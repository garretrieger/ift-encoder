# Reworking Initial Font Handling in the IFT Encoder

Author: Garret Rieger  
Date: Nov 19th, 2026

## Overview & Motivation

Currently, the segmenter and compiler treat the initial font as a single subset definition whose full codepoint and glyph closure (`GlyphClosureCache::ExpandClosure`) is recomputed whenever glyphs are moved into the initial font, marking all resulting codepoints, features, and glyphs as always-true (`Node::InitFont`) in `DependencyClosure`:

* **Premature condition weakening**: If `'f'` is moved to the initial font and `'i'` is not, the condition for the `'fi'` ligature glyph simplifies from `(s_f AND s_i)` to `(s_i)`, causing clients rendering text with `'i'` (and no `'f'`) to unnecessarily download `'fi'`. Similarly, when optional layout feature segments are enabled (`generate_feature_segments = true`), moving `'a'..'z'` simplifies every `(s_a AND s_smcp)` condition to `(s_smcp)`, forcing pages that enable `smcp` on a heading or non-Latin script to download small-cap glyphs for all moved Latin letters.
* **Unintended closure expansion**: Calling `ExpandClosure` during init-font moves reverse-maps moved glyphs back to all associated `cmap` codepoints and transitively closures GSUB/component dependencies.
* **Expensive incremental init-font re-analysis**: Every non-batch init-font move currently triggers `SegmentationContext::ReassignInitSubset` and `DependencyClosure::InitFontChanged`, re-propagating conditions across the dependency graph and recomputing modified/colliding patches in `CandidateMerge::ComputeInitFontCostDelta`.

Instead, it is valid in IFT to place a glyph's outline data into the initial font while keeping its triggering segment active in `IFTX` activation conditions (and, when standalone rendering of a codepoint needs no remaining patches, also placing that codepoint into `initial_codepoints` so `cmap`/`hmtx`/`GPOS`/`GSUB` are present in the initial font for 0-RTT rendering while `plan.segments` retains the segment for conjunctive `IFTX` conditions).

> Note: Fonts with no glyph-keyed outline data at all (e.g. pure `CBDT`/`COLR` fonts) should skip glyph-keyed segmentation completely; that will be handled in a separate effort.

---

## Architecture & Design

### 1. Frozen Closure Init Subset vs. Moved Init Glyphs & Subset (`InitialFont`)

`RequestedSegmentationInformation` previously conflated the dependency-graph closure baseline with glyphs/codepoints moved during optimization. This state is encapsulated in [`InitialFont`](../ift/encoder/initial_font.h) (`init_font_` on `RequestedSegmentationInformation`), separating input closure, unsegmented closure fallback, and mutable output sets:

1. **Closed Input Baseline (`InitialFont::closed_input_definition_`)**:
   * Computed once during `RequestedSegmentationInformation::Create` by applying `AddInitSubsetDefaults` and `GlyphClosureCache::ExpandClosure` to `SegmenterConfig.initial_segment`.
   * Codepoints in `closed_input_definition_.codepoints` are subtracted from input `segments_` at initialization as today.
   * Used exclusively (along with `NonClosedInputGlyphs()`) by `DependencyGraph`, `DependencyClosure`, and `GlyphClosureCache::SegmentClosure`, and **never modified** during optimization moves.
2. **Glyph-Keyed Initial Font Glyphs (`InitialFont::glyph_keyed_glyphs_`)**:
   * Initialized to `closed_input_definition_.gids`.
   * Empty outline glyphs are handled via the `empty_glyphs_init_font` approach: in `IFT` target mode, `NonInitFontNonEmptyGlyphs()` (`NonInitFontGlyphs() \ empty_glyphs_`) excludes unmoved empty glyphs from `TargetGlyphs()` and `ValidateSegmentation`, while `InitFontMoveTargetGlyphs()` retains empty glyphs during init-font moves until `ExcludeEmptyGlyphs()` is called (when an opted-in strategy uses init-font merging).
   * During `InitializeSegmentationContext` (`always_true_glyphs`) and the init-font move phase, moved glyphs are unioned directly into `glyph_keyed_glyphs_` (and removed from `non_init_font_glyphs_` and `non_init_font_non_empty_glyphs_`) via `MoveGlyphsToInitFont` (`InitialFont::AddGlyphKeyedGlyphs`) with **no closure expansion** and **no `DependencyClosure` update**.
   * `GlyphSegmentation::ToSegmentationPlanProto()` emits `InitialFontGlyphClosure()` into `SegmentationPlan.initial_glyphs`.
3. **Table-Keyed Initial Font Sets (`InitialFont::table_keyed_codepoints_`, `table_keyed_features_`, & `unsegmented_definition_`)**:
   * Initialized from `closed_input_definition_` and extended during the init-font move phase with eligible moved codepoints/features (without closure expansion) via `AddToInitFontSegment` (`InitialFont::AddToTableKeyed`).
   * `unsegmented_definition_` (`InitialFont::UnsegmentedDefinition()`) tracks `closed_input_definition_` plus any segments cleared by `ClearUnusedInitSegments` (`InitialFont::AddClearedSegment`) so `GlyphClosureCache::ComputeExceptSegment` matches `FullDefinition()`.
   * Subtracted from `AddTableKeyedSegments` and emitted as `initial_codepoints` / `initial_features` in `SegmentationPlan` (with defaults stripped via `TableKeyedDefinitionWithoutDefaults()` / `InitFontSegmentWithoutDefaults()`).
   * Adding codepoints/features to `InitialFont::table_keyed_*` does **not** remove them from `segments_[s]` as long as `s` still appears in at least one target glyph condition; segments with no remaining target glyph conditions are cleared via `ClearUnusedInitSegments` so dead segments are omitted from `plan.segments`.

### 2. Init Font Move Phase (`Merger` & `CandidateMerge`)

#### Patch Moves
Because patches in `GlyphGroupings::ConditionsAndGlyphs()` are mutually glyph-disjoint and `DependencyClosure` + `segments_` remain unchanged during patch moves, moving a patch `(condition, glyphs)` into `InitialFont::glyph_keyed_glyphs_` simply drops that patch without modifying any other patch's glyph set or activation condition:

* **Best-case cost delta**:
  $$\Delta \text{cost}_{\text{best}} = \text{best\_case\_reduction} \times \text{size}(\text{glyphs}) - P_m(\text{condition}) \times (\text{size}(\text{glyphs}) + k)$$
* **Exact cost delta**:
  $$\Delta \text{cost} = \Delta \text{init\_size} - P_m(\text{condition}) \times (\text{size}(\text{glyphs}) + k)$$
  where $\Delta \text{init\_size} = \max(0, \text{size}(G_{\text{init}} \cup \text{glyphs}) - \text{size}(G_{\text{init}}))$ (tracked via `smallest_size_increases[glyphs]` to dampen Brotli noise).
* Applying a move removes `glyphs` from `NonInitFontGlyphs()`, `GlyphConditionSet`, and `GlyphGroupings` directly without running `ExpandClosure` or `DependencyClosure::InitFontChanged`.

#### Codepoint & Feature Moves
During the init-font move phase, after glyph patches (including high-probability empty-glyph patches kept in `InitFontMoveTargetGlyphs()`) have been moved to `InitialFont::glyph_keyed_glyphs_` and `ExcludeEmptyGlyphs()` has removed remaining low-probability empty glyphs, segments are checked for promotion into `InitialFont::table_keyed_*`:

1. **Candidate filter**: Consider only segments `s` that:
   * Belong to an opted-in `ProbabilityProfile` (`init_font_merge_threshold.has_value()`) and pass that profile's probability threshold (`Merger::InitFontApplyProbabilityThreshold(profile_index)`), **and**
   * Appear in the condition (`DependencyClosure::AllGlyphConditions()`) of at least one glyph in `InitialFont::glyph_keyed_glyphs_`.
2. **Single-segment satisfaction check**: Add `s`'s codepoints and features to `InitialFont::table_keyed_*` **iff** no remaining patch condition `c` in `GlyphGroupings::ConditionsAndGlyphs()` is satisfied by `s` alone:
   ```cpp
   bool IsSatisfiedBySingleSegment(const ActivationCondition& c, segment_index_t s) {
     for (const SegmentSet& clause : c.conditions()) {
       if (!clause.contains(s)) return false;
     }
     return !c.conditions().empty();
   }
   ```
   This uniformly excludes segments that still have an unmoved exclusive patch `if (s)` or an unmoved disjunctive patch `if (s OR s_2)` (avoiding vacuously empty `ExclusiveGlyphs(s)` when glyphs are shared), while allowing segments whose only remaining conditions are conjunctive (such as `s_f AND s_i` or `s_a AND s_smcp`) to move into `InitialFont::table_keyed_*`.
3. **Segment cleanup**: After init-font moving finishes across all profiles, any segment `s` whose definition is a subset of `InitialFont::TableKeyedDefinition()` and which has `TriggeringSegmentToConditions(s).empty()` is cleared via `ClearUnusedInitSegments`; segments still referenced by target glyph conditions remain intact for `AllPairsMerger` and `IFTX` condition compilation.

### 3. Merging Segments Without Exclusive Glyphs (`AllPairsMerger` & `Merger`)

Segments whose exclusive glyphs moved to `InitialFont::glyph_keyed_glyphs_` remain active if they still appear in target glyph conditions (e.g. `s_a..s_z` in `(s_a AND s_smcp)`). Expanding `AllPairsMerger` to merge these segments avoids leaving dozens of 1-glyph feature/ligature patches unmerged, while preserving two invariants:

* **Partitioned pairwise candidate generation**: Active segments in `AllPairsMerger` (`s < optimization_cutoff_segment_` with `!TriggeringSegmentToConditions(s).empty()`) are partitioned into two probability-ordered sets:
  1. `active_segments_by_prob_`: segments with `!ExclusiveGlyphs(s).empty()` (paired with each other as today, preserving `BestCaseInertProbabilityThreshold` pruning and Phase 2 `min_group_size` iteration).
  2. `active_no_exclusive_segments_by_prob_`: segments with `ExclusiveGlyphs(s).empty()` (paired **only with each other**).
  Keeping the two partitions separate prevents pairing moved 0-RTT init-font segments with unmoved exclusive-patch segments — avoiding both Stage-1 best-case heap false positives (where `source_count == 1` in `ComputeCostDelta<true>` skips the unmoved exclusive patch's probability-increase penalty) and breaking 0-RTT rendering of init-font codepoints.
* **Composite candidate restriction**: `AllPairsMerger::QueueCompositeCandidates` (as well as `Merger::CollectCompositeCandidateMerges` and `Merger::TryMergingACompositeCondition`) skips candidate `triggering_segments` sets that mix segments with exclusive glyphs and segments without exclusive glyphs.

### 4. Compiler & Total Costs Updates

`Compiler` explicitly separates non-outline subsetting from outline table population in mixed mode:

* **Non-outline tables (`init_subset_`)**: `initial_codepoints`, `initial_features`, `initial_design_space`, and `initial_segments` (plus `AddInitSubsetDefaults` and `gid_count - 1` for `maxp.numGlyphs` retention) are subsetted with `hb_subset` closure as usual, determining glyph presence in `cmap`, `hmtx`, `GSUB`, `GPOS`, `COLR`, etc. across all nodes of the table-keyed graph.
* **Outline tables (`init_glyphs_`)**: When explicit `init_glyphs_` is configured in mixed mode (`SetInitGlyphs`), the base outline tables (`glyf`/`loca`, `gvar`, `CFF `, `CFF2`) are populated for `*init_glyphs_` without layout, `cmap`, `MATH`, `COLR`, or `VARC` closure via an outline-only `hb_subset` helper (and `GenerateBaseCff2` for `CFF2`), then transplanted onto the `CutSubsetFaceBuilder` face:
  * **Outline `hb_subset` configuration**: Populate only `HB_SUBSET_SETS_GLYPH_INDEX` with `*init_glyphs_`, drop non-outline closure tables (`cmap`, `GSUB`, `GPOS`, `GDEF`, `MATH`, `COLR`, `VARC`) via `HB_SUBSET_SETS_DROP_TABLE_TAG`, and set `HB_SUBSET_FLAGS_NO_LAYOUT_CLOSURE | HB_SUBSET_FLAGS_RETAIN_GIDS | HB_SUBSET_FLAGS_RETAIN_NUM_GLYPHS | HB_SUBSET_FLAGS_PASSTHROUGH_UNRECOGNIZED | HB_SUBSET_FLAGS_DESUBROUTINIZE` (plus `HB_SUBSET_FLAGS_IFTB_REQUIREMENTS` when `force_long_loca_and_gvar_` is set, and `HB_SUBSET_FLAGS_NOTDEF_OUTLINE` only when `init_glyphs_->contains(0)`). Composite glyph components (`_glyf_add_gid_and_children` in `glyf` and `seac` in `CFF `) remain closured by `hb_subset`, which is needed for 0-RTT rendering of composite glyphs in `init_glyphs_` and idempotent with later `IFTX` patches.
  * **`glyf` + `loca` (+ `head.indexToLocFormat`)**: Run the outline `hb_subset` on `full_face` (with `def.design_space`), replace `glyf` and `loca` on `result`, and sync `head`'s `indexToLocFormat` field from the outline subset's `head` so short/long `loca` cannot drift.
  * **`gvar`**: In `GenerateBaseGvar`, keep step 1 (`Instance` on `full_face` retaining all glyphs so shared tuples match `GlyphKeyedDiff`), and run the outline-only `hb_subset` (with empty `design_space`) in step 2 to extract `gvar`.
  * **`CFF `**: Extract `CFF ` directly from the outline `hb_subset` of `full_face` (desubroutinized, charstrings packed at the end, and `FDSelect`/`FDArray`/`Charset` retained across all GIDs via `HB_SUBSET_FLAGS_RETAIN_NUM_GLYPHS`).
  * **`CFF2`**: Pass `*init_glyphs_` directly to `GenerateCharStringsTable` in `GenerateBaseCff2` instead of running `CreateSubsetPlan`.
  * When `!IsMixedMode()` or when legacy callers/tests use `SetInitSubset` without `SetInitGlyphs`, existing closure-based outline subsetting is preserved.
* **`ClosureGlyphSegmenter::TotalCosts`**: Computes `init_font_size` in `IFT` mode by cutting the initial font with the same separated non-outline `InitialFontSegment()` + outline-only `InitialFontGlyphClosure()` subsetting and WOFF2-encoding at quality 11, rather than calling `CandidateMerge::Woff2SizeOf` with full `hb_subset` closure over `InitialFontSegment()`.

---

## Implementation Plan / TODO

- [x] **Commit 1: Separate `init_subset_` and non-layout-closured `init_glyphs_` in `Compiler`.**
  * **Files**: `ift/encoder/compiler.{h,cc}`, `ift/config/config_compiler.cc`, `ift/encoder/compiler_test.cc`, `ift/integration_test.cc`.
  * **Changes**:
    * Add `Compiler::SetInitGlyphs(const GlyphSet&)` (`std::optional<GlyphSet> init_glyphs_`) and wire `ConfigCompiler::Configure` (`Compiler` overload) to pass `plan.initial_glyphs()` via `SetInitGlyphs` instead of `init_subset.gids` when `plan` has glyph patches / `initial_glyphs`.
    * Update `Compiler::FullyExpandedSubset` to union `*init_glyphs_` when present.
    * Update `Compiler::CutSubset`, `GenerateBaseGvar`, `GenerateBaseCff2`, and add an outline-only `hb_subset` helper (`HB_SUBSET_FLAGS_NO_LAYOUT_CLOSURE | HB_SUBSET_FLAGS_RETAIN_GIDS | HB_SUBSET_FLAGS_RETAIN_NUM_GLYPHS`, dropping `cmap`/`GSUB`/`GPOS`/`GDEF`/`MATH`/`COLR`/`VARC`) to extract `glyf`/`loca` (+ `head.indexToLocFormat` sync), `gvar`, and `CFF ` when `IsMixedMode() && init_glyphs_.has_value()`, while non-outline tables continue using `hb_subset` closure on `node_subset`.
    * Add unit and Fontations client integration tests covering `glyf`, `gvar`, `CFF `, and `CFF2` where `'f'` and `'i'` are in `initial_codepoints` and `gid('f')`, `gid('i')` are in `initial_glyphs`, verifying that `gid('fi')` is excluded from the initial font's outline tables and loaded on demand via `IFTX` when extending to `"fi"`, while `"f"` requires 0 glyph-keyed and 0 table-keyed patches.

- [x] **Commit 2: Introduce `InitialFont` and split frozen `closed_input_definition_` from mutable table-keyed/glyph-keyed sets.**
  * **Files**: `ift/encoder/initial_font.{h,cc}`, `ift/encoder/requested_segmentation_information.{h,cc}`, `ift/encoder/requested_segmentation_information_test.cc`, `ift/encoder/segmentation_context.{h,cc}`, `ift/encoder/glyph_groupings.cc`, `ift/encoder/glyph_segmentation.cc`, `ift/dep_graph/dependency_graph.cc`, `ift/encoder/dependency_closure.{h,cc}`, `ift/encoder/glyph_closure_cache.cc`, `ift/encoder/closure_glyph_segmenter_test.cc`, `ift/encoder/BUILD`, `docs/init_font_handling.md`.
  * **Changes**:
    * Extract [`InitialFont`](../ift/encoder/initial_font.h) held as `init_font_` on `RequestedSegmentationInformation`, encapsulating `closed_input_definition_`, `unsegmented_definition_`, `table_keyed_codepoints_`, `table_keyed_features_`, and `glyph_keyed_glyphs_`, plus `NonClosedInputGlyphs()` (used by `DependencyGraph`, `DependencyClosure`, `GlyphClosureCache::SegmentClosure`, and `ComputeExceptSegment`) alongside mutable non-closured moves (`MoveGlyphsToInitFont`, `AddToInitFontSegment`, `ClearUnusedInitSegments`, `ReplayInitFontMoves`) and existing empty-glyph handling (`NonInitFontNonEmptyGlyphs()`, `InitFontMoveTargetGlyphs()`, `ExcludeEmptyGlyphs()`).
    * Update `SegmentationContext::WithSameSettings` so `ValidateIncrementalGroupings` preserves `InitialFont` state via `ReplayInitFontMoves`.
    * Add `SegmentationContext::{MoveGlyphsToInitFont,AddToInitFontSegment,ClearUnusedInitSegments}`, switch initial `AlwaysTrueGlyphs()` handling to `MoveGlyphsToInitFont`, and emit `InitialFontGlyphClosure()` in `GlyphSegmentation::ToSegmentationPlanProto()`.
    * Add unit tests in `RequestedSegmentationInformationTest` and `ClosureGlyphSegmenterTest`, and update `AlwaysTrueGlyphs_InitialSegmentMovedToInitFont` for `InitialFontSegment().gids` vs `InitialFontGlyphClosure()` / `plan.initial_glyphs()`.

- [x] **Commit 3: Rework `MoveSegmentsToInitFont` and `CandidateMerge` init-font cost delta to move patches and eligible segments without closure expansion.**
  * **Files**: `ift/encoder/initial_font.h`, `ift/encoder/requested_segmentation_information_test.cc`, `ift/encoder/candidate_merge.{h,cc}`, `ift/encoder/candidate_merge_test.cc`, `ift/encoder/merger.{h,cc}`, `ift/encoder/segmentation_context.{h,cc}`, `ift/encoder/closure_glyph_segmenter.cc`, `ift/encoder/closure_glyph_segmenter_test.cc`, `docs/init_font_handling.md`.
  * **Changes**:
    * Simplify `CandidateMerge::ComputeBestCaseInitFontCostDelta` and `CandidateMerge::ComputeInitFontCostDelta` to evaluate single-patch removal (`init_increase - P_m(c) * (size(G) + k)`) without `ComputeInitFontGlyphDelta` / `ExpandClosure` or cross-patch condition rewriting.
    * Update `Merger::ApplyInitFontMove` / `SegmentationContext` to remove moved `glyphs` directly from `NonInitFontGlyphs()`, `GlyphConditionSet`, and `GlyphGroupings` (and refresh `inert_segments_`) without calling `DependencyClosure::InitFontChanged`.
    * After patch moves for opted-in profiles, promote segments in `InitFontApplyProbabilityThreshold(profile_index)` that appear in the dependency-graph condition of at least one glyph in `InitialFont::glyph_keyed_glyphs_` into `InitialFont::table_keyed_*` iff no remaining patch condition in `GlyphGroupings` is satisfied by `s` alone; clear moved segments that have `TriggeringSegmentToConditions(s).empty()`.
    * Update existing init-font merging tests and add unit tests verifying: (a) moving `'f'` leaves `'fi'` as `(s_f AND s_i)`, (b) moving both `'f'` and `'i'` moves both codepoints to `InitialFontSegment()` while keeping `s_f` and `s_i` active for `p_fi`, (c) shared unmoved disjunctive `(s_1 OR s_2)` glyphs prevent moving `s_1`/`s_2` to `InitialFontSegment()`, and (d) high-probability empty-glyph segments (e.g. `U+0020`) move to `InitialFontSegment()` while low-probability ones and `GPOS`-only optional features do not.

- [x] **Commit 4: Expand `AllPairsMerger` and composite candidate selection to support zero-exclusive-glyph segments.**
  * **Files**: `ift/encoder/all_pairs_merger.{h,cc}`, `ift/encoder/merger.{h,cc}`, `ift/encoder/activation_condition.cc`, `ift/dep_graph/dependency_graph.cc`, `ift/encoder/dependency_closure.{h,cc}`, `ift/encoder/segmentation_context.{h,cc}`, `ift/encoder/closure_glyph_segmenter.cc`, `ift/encoder/closure_glyph_segmenter_test.cc`, `docs/init_font_handling.md`.
  * **Changes**:
    * Update `AllPairsMerger::UpdateActiveSegment` and `RefreshDirtySegments` to keep segments with `ExclusiveGlyphs(s).empty()` and `!TriggeringSegmentToConditions(s).empty()` active in a dedicated `active_no_exclusive_segments_by_prob_` set, queuing pairwise candidates strictly within each partition (`exclusive` × `exclusive` and `no-exclusive` × `no-exclusive`).
    * Filter out composite `triggering_segments` sets that mix `exclusive` and `no-exclusive` segments in `AllPairsMerger::QueueCompositeCandidates`, `Merger::CollectCompositeCandidateMerges`, and `Merger::TryMergingACompositeCondition`.
    * Add tests in `ClosureGlyphSegmenterTest` verifying that moved init-font segments with optional feature glyphs (e.g. `'a'..'z'` with `smcp`) merge their zero-exclusive-glyph segments via `AllPairsMerger` to consolidate `smcp` patches without merging into unmoved exclusive-patch segments.

- [ ] **Commit 5: Update `TotalCosts`, end-to-end tests, and documentation.**
  * **Files**: `ift/encoder/compiler.{h,cc}`, `ift/encoder/closure_glyph_segmenter.cc`, `ift/encoder/closure_glyph_segmenter_test.cc`, `ift/integration_test.cc`, `docs/segmenter.md`, `docs/compiler.md`, `docs/init_font_handling.md`.
  * **Changes**:
    * Update `ClosureGlyphSegmenter::TotalCosts` to compute `init_font_size` using separated non-outline `InitialFontSegment()` + outline-only `InitialFontGlyphClosure()` (non-layout-closured) WOFF2 sizing (sharing a helper with `Compiler`).
    * Add an end-to-end `SegmenterConfigUtil::RunSegmenter` → `ConfigCompiler::Configure` → `Compiler::Compile` → Fontations `Extend` integration test verifying 0-RTT initial-font rendering on moved ASCII codepoints alongside on-demand `IFTX` loading of unmoved ligatures and optional `smcp` patches.
    * Update `docs/segmenter.md` ("Initial Font Merging") and `docs/compiler.md` ("Initial Subset"), and mark `docs/init_font_handling.md` complete (or retire it into `docs/segmenter.md`).
