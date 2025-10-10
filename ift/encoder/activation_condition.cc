#include "ift/encoder/activation_condition.h"

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "common/int_set.h"
#include "common/sparse_bit_set.h"
#include "common/try.h"
#include "ift/encoder/glyph_partition.h"
#include "ift/encoder/subset_definition.h"
#include "ift/encoder/types.h"
#include "ift/proto/patch_encoding.h"
#include "ift/proto/patch_map.h"

using absl::btree_set;
using absl::flat_hash_map;
using absl::Span;
using absl::Status;
using absl::StatusOr;
using absl::StrCat;
using common::IntSet;
using common::GlyphSet;
using common::SegmentSet;
using common::SparseBitSet;
using common::CodepointSet;
using ift::freq::ProbabilityCalculator;
using ift::proto::PatchEncoding;
using ift::proto::PatchMap;

namespace ift::encoder {

ActivationCondition ActivationCondition::exclusive_segment(
    segment_index_t index, patch_id_t activated) {
  ActivationCondition condition;
  condition.activated_ = {activated};
  condition.conditions_ = {{index}};
  condition.is_exclusive_ = true;
  return condition;
}

ActivationCondition ActivationCondition::and_segments(
    const SegmentSet& segments, patch_id_t activated) {
  ActivationCondition conditions;
  conditions.activated_ = {activated};

  for (auto id : segments) {
    conditions.conditions_.push_back(SegmentSet{id});
  }

  return conditions;
}

ActivationCondition ActivationCondition::or_segments(const SegmentSet& segments,
                                                     patch_id_t activated,
                                                     bool is_fallback) {
  ActivationCondition conditions;
  conditions.activated_ = {activated};
  conditions.conditions_.push_back(segments);
  conditions.is_fallback_ = is_fallback;

  return conditions;
}

ActivationCondition ActivationCondition::composite_condition(
    absl::Span<const SegmentSet> groups, patch_id_t activated) {
  ActivationCondition conditions;
  conditions.activated_ = {activated};
  for (const auto& group : groups) {
    conditions.conditions_.push_back(group);
  }

  return conditions;
}

std::string ActivationCondition::ToString() const {
  std::stringstream out;
  out << "if (";
  bool first = true;
  for (const auto& set : conditions()) {
    if (!first) {
      out << " AND ";
    } else {
      first = false;
    }

    if (set.size() > 1) {
      out << "(";
    }
    bool first_inner = true;
    for (uint32_t id : set) {
      if (!first_inner) {
        out << " OR ";
      } else {
        first_inner = false;
      }
      out << "s" << id;
    }
    if (set.size() > 1) {
      out << ")";
    }
  }
  out << ") then p" << activated();
  return out.str();
}

bool ActivationCondition::operator<(const ActivationCondition& other) const {
  if (conditions_.size() != other.conditions_.size()) {
    return conditions_.size() < other.conditions_.size();
  }

  auto a = conditions_.begin();
  auto b = other.conditions_.begin();
  while (a != conditions_.end() && b != other.conditions_.end()) {
    if (a->size() != b->size()) {
      return a->size() < b->size();
    }

    auto aa = a->begin();
    auto bb = b->begin();
    while (aa != a->end() && bb != b->end()) {
      if (*aa != *bb) {
        return *aa < *bb;
      }
      aa++;
      bb++;
    }

    a++;
    b++;
  }

  if (activated_ != other.activated_) {
    return activated_ < other.activated_;
  }

  if (is_exclusive_ != other.is_exclusive_) {
    return is_exclusive_;
  }

  if (is_fallback_ != other.is_fallback_) {
    return !is_fallback_;
  }

  // These two are equal
  return false;
}

template <typename ProtoType>
ProtoType ToSetProto(const IntSet& set) {
  ProtoType values;
  for (uint32_t v : set) {
    values.add_values(v);
  }
  return values;
}

ActivationConditionProto ActivationCondition::ToConfigProto() const {
  ActivationConditionProto proto;

  for (const auto& ss : conditions()) {
    *proto.add_required_segments() = ToSetProto<SegmentsProto>(ss);
  }
  proto.set_activated_patch(activated());

  return proto;
}

void MakeIgnored(PatchMap::Entry& entry, patch_id_t& last_patch_id) {
  entry.ignored = true;
  // patch id for ignored entries doesn't matter, use last + 1 to minimize
  // encoding size.
  entry.patch_indices.clear();
  entry.patch_indices.push_back(++last_patch_id);
}

patch_id_t MapTo(PatchMap::Entry& entry, patch_id_t new_patch_id,
                 Span<const patch_id_t> prefetches) {
  entry.ignored = false;
  entry.patch_indices.clear();
  entry.patch_indices.push_back(new_patch_id);
  entry.patch_indices.insert(entry.patch_indices.end(), prefetches.begin(),
                             prefetches.end());
  return entry.patch_indices.back();
}

static unsigned CodepointEncodingSize(const SubsetDefinition& segment) {
  return SparseBitSet::Encode(segment.codepoints).size();
}

static segment_index_t MaxSegmentIndex(const btree_set<ActivationCondition>& conditions) {
  segment_index_t max = 0;
  for (const auto& c : conditions) {
    for (segment_index_t s : c.TriggeringSegments()) {
      if (s > max) {
        max = s;
      }
    }
  }
  return max;
}

static bool MustUseSharedSegments(
  const flat_hash_map<segment_index_t, SubsetDefinition>& segments,
  const ActivationCondition& condition
) {
  // Any conjunctive conditions, or conditions with at least one non codepoint
  // only segment must always used shared segments.
  if (condition.conditions().size() > 1) {
    return true;
  }

  SegmentSet condition_segments = condition.TriggeringSegments();
  for (segment_index_t s : condition_segments) {
    const SubsetDefinition& def = segments.at(s);
    if (!def.design_space.empty() || !def.feature_tags.empty()) {
      return true;
    }
  }

  return false;
}

static CodepointSet CodepointsFor(
  const SegmentSet& condition_segments,
  const flat_hash_map<segment_index_t, SubsetDefinition>& segments
) {
  CodepointSet all;
  for (segment_index_t s : condition_segments) {
    const SubsetDefinition& def = segments.at(s);
    all.union_set(def.codepoints);
  }
  return all;
}

static Status CompareDuplicationCosts(
  const btree_set<ActivationCondition>& conditions,
  const flat_hash_map<segment_index_t, SubsetDefinition>& segments,
  const GlyphPartition& segment_partition,
  const SegmentSet& shared_reps,
  SegmentSet& segments_to_duplicate
) {
  flat_hash_map<segment_index_t, unsigned> rep_to_duplicated_size;
  flat_hash_map<segment_index_t, unsigned> rep_to_shared_size;

  for (const auto& c : conditions) {
    segment_index_t first = *(c.conditions().begin()->begin());
    segment_index_t rep = TRY(segment_partition.Find(first));
    if (shared_reps.contains(rep)) {
      continue;
    }

    SegmentSet condition_segments = c.TriggeringSegments();
    CodepointSet all_codepoints = CodepointsFor(condition_segments, segments);

    rep_to_duplicated_size[rep] += SparseBitSet::Encode(all_codepoints).size();
    // 1 byte child count + 3 byte child entry index per segment
    rep_to_shared_size[rep] += 1 + condition_segments.size() * 3;
  }

  for (auto& [rep, shared_size] : rep_to_shared_size) {
    for (segment_index_t segment : TRY(segment_partition.GlyphsFor(rep))) {
      const CodepointSet& codepoints = segments.at(segment).codepoints;
      // add cost of the child entry, 1 byte format + size of the sparse bit set.
      shared_size += 1 + SparseBitSet::Encode(codepoints).size();
    }
  }

  for (const auto& [rep, shared_size] : rep_to_shared_size) {
    unsigned duplicated_size = rep_to_duplicated_size.at(rep);
    if (duplicated_size < shared_size) {
      segments_to_duplicate.union_set(TRY(segment_partition.GlyphsFor(rep)));
    }
  }

  return absl::OkStatus();
}

// Analyze a set of activation conditions and decide which segments should
// not be shared via the child index mechanism due to it being cheaper to
// just encode duplicate copies of the segments subset definition.
static Status FindSegmentsToDuplicate(
  const btree_set<ActivationCondition>& conditions,
  const flat_hash_map<segment_index_t, SubsetDefinition>& segments,
  SegmentSet& segments_to_duplicate
) {
  // TODO XXXX what about the fallback segment, that will pull everything
  //           into one partitition...
  segments_to_duplicate.clear();
  if (conditions.empty()) {
    return absl::OkStatus();
  }

  // Group into partitions of connected segments.
  GlyphPartition segment_partition(MaxSegmentIndex(conditions));
  for (const auto& c : conditions) {
    GlyphSet set;
    set.swap(c.TriggeringSegments());
    TRYV(segment_partition.Union(set));
  }

  // Then if at least one segment in a partition must be shared then all
  // of them have to be shared. Track partition representatives that are shared.
  SegmentSet shared_reps;
  for (const auto& c : conditions) {
    segment_index_t first = *(c.conditions().begin()->begin());
    segment_index_t rep = TRY(segment_partition.Find(first));
    if (shared_reps.contains(rep)) {
      // Already marked shared.
      continue;
    }

    if (MustUseSharedSegments(segments, c)) {
      shared_reps.insert(rep);
    }
  }

  return CompareDuplicationCosts(conditions, segments, segment_partition, shared_reps, segments_to_duplicate);
}

StatusOr<std::vector<PatchMap::Entry>>
ActivationCondition::ActivationConditionsToPatchMapEntries(
    Span<const ActivationCondition> conditions,
    const flat_hash_map<segment_index_t, SubsetDefinition>& segments) {
  std::vector<PatchMap::Entry> entries;
  if (conditions.empty()) {
    return entries;
  }

  // The conditions list describes what the patch map should do, here
  // we need to convert that into an equivalent list of encoder condition
  // entries.
  //
  // To minimize encoded size we can reuse set definitions in later entries
  // via the copy indices mechanism. The conditions are evaluated in three
  // phases to successively build up a set of common entries which can be reused
  // by later ones.
  //
  // Tracks the list of conditions which have not yet been placed in a map
  // entry.
  btree_set<ActivationCondition> remaining_conditions;
  remaining_conditions.insert(conditions.begin(), conditions.end());

  // For small segments, it can be more costly to reuse (share) the subset definition
  // instead of just encoding it directly when needed, this computes the set
  // of segments which we must be shared.
  SegmentSet segments_to_duplicate;
  TRYV(FindSegmentsToDuplicate(remaining_conditions, segments, segments_to_duplicate));

  // Phase 1 generate the base entries, there should be one for each
  // unique glyph segment that is referenced in at least one condition.
  // the conditions will refer back to these base entries via copy indices
  //
  // Each base entry can be used to map one condition as well.
  flat_hash_map<uint32_t, uint32_t> segment_id_to_entry_index;
  uint32_t next_entry_index = 0;
  patch_id_t last_patch_id = 0;
  for (auto condition = remaining_conditions.begin();
       condition != remaining_conditions.end();) {
    bool remove = false;
    for (uint32_t segment_id : condition->TriggeringSegments()) {
      if (segment_id_to_entry_index.contains(segment_id)) {
        continue;
      }

      if (!condition->IsUnitary() && segments_to_duplicate.contains(segment_id)) {
        // we have decided to duplicate instead of share this segment so skip
        // encoding it here, unless this is a non-composite condition in which
        // case this is not a case of sharing and we will need to encode it.
        continue;
      }

      auto original = segments.find(segment_id);
      if (original == segments.end()) {
        return absl::InvalidArgumentError(
            StrCat("Codepoint segment ", segment_id, " not found."));
      }
      const auto& original_def = original->second;

      std::vector<PatchMap::Entry> sub_entries =
          // Activated patch ID will be assigned after this step, so just use
          // empty array as a place holder
          original_def.ToEntries(condition->encoding_, last_patch_id,
                                 entries.size(), {});
      auto& sub_entry = sub_entries.back();

      last_patch_id = sub_entry.patch_indices.back();
      if (condition->IsUnitary()) {
        // this condition can use this entry to map itself. Update the entries
        // mapped patch id.
        last_patch_id =
            MapTo(sub_entry, condition->activated(), condition->prefetches());
        remove = true;
      }

      entries.insert(entries.end(), sub_entries.begin(), sub_entries.end());
      next_entry_index = entries.size();
      segment_id_to_entry_index[segment_id] = next_entry_index - 1;
    }

    if (remove) {
      condition = remaining_conditions.erase(condition);
    } else {
      ++condition;
    }
  }

  // Phase 2 generate entries for all groups of patches reusing the base entries
  // written in phase one. When writing an entry if the triggering group is the
  // only one in the condition then that condition can utilize the entry (just
  // like in Phase 1).
  flat_hash_map<IntSet, uint32_t> segment_group_to_entry_index;
  for (auto condition = remaining_conditions.begin();
       condition != remaining_conditions.end();) {
    bool remove = false;

    for (const auto& group : condition->conditions()) {
      if (group.size() <= 1 || segment_group_to_entry_index.contains(group)) {
        // don't handle groups of size one, those will just reference the base
        // entry directly.
        continue;
      }

      PatchMap::Entry entry;
      entry.encoding = condition->encoding_;

      for (uint32_t segment_id : group) {
        if (!segments_to_duplicate.contains(segment_id)) {
          entry.coverage.conjunctive = false;  // ... OR ...
          auto entry_index = segment_id_to_entry_index.find(segment_id);
          if (entry_index == segment_id_to_entry_index.end()) {
            return absl::InternalError(
                StrCat("entry for segment_id = ", segment_id,
                     " was not previously created."));
          }
          entry.coverage.child_indices.insert(entry_index->second);
        } else {
          // We're not reusing a shared entry so just union the def into the
          // entry's definition. The processing in FindSegmentsToDuplicate() ensures
          // this is done consistently for all segments in a condition, and that
          // all segments involved are codepoint only.
          const SubsetDefinition& def = segments.at(segment_id);
          entry.coverage.codepoints.union_set(def.codepoints);
        }
      }

      if (condition->conditions().size() == 1) {
        last_patch_id =
            MapTo(entry, condition->activated(), condition->prefetches());
        remove = true;
      } else {
        MakeIgnored(entry, last_patch_id);
      }

      entries.push_back(entry);
      segment_group_to_entry_index[group] = next_entry_index++;
    }

    if (remove) {
      condition = remaining_conditions.erase(condition);
    } else {
      ++condition;
    }
  }

  // Phase 3 for any remaining conditions create the actual entries utilizing
  // the groups (phase 2) and base entries (phase 1) as needed
  for (auto condition = remaining_conditions.begin();
       condition != remaining_conditions.end(); condition++) {
    PatchMap::Entry entry;
    entry.encoding = condition->encoding_;
    entry.coverage.conjunctive = true;  // ... AND ...

    // We don't need to checked for duplicated segments here since conjunctive conditions
    // will never use them.
    for (const auto& group : condition->conditions()) {
      if (group.size() == 1) {
        entry.coverage.child_indices.insert(
            segment_id_to_entry_index[*group.begin()]);
        continue;
      }

      entry.coverage.child_indices.insert(segment_group_to_entry_index[group]);
    }

    last_patch_id =
        MapTo(entry, condition->activated(), condition->prefetches());
    entries.push_back(entry);
  }

  return entries;
}

StatusOr<double> ActivationCondition::Probability(
    Span<const Segment> segments,
    const ProbabilityCalculator& calculator) const {
  // This calculation makes the assumption that segments are all disjoint.
  // Disjointess of the segment list is enforced in the initialization
  // of segmentation context.

  std::vector<const Segment*> conjunctive_segments;
  bool is_conjunctive = conditions_.size() > 1;
  for (const auto& segment_set : conditions_) {
    if (is_conjunctive && segment_set.size() != 1) {
      // Composite conditions (eg. (a or b) and (c or d)) may have repeated
      // segments in each conjunctive group (eg. (a or b) and (a or d)) which
      // requires special analysis to correctly determine probability. For our
      // current use cases we don't need to support this.
      return absl::UnimplementedError(
          "Calculating probability of composite conditions is not "
          "supported.");
    }

    if (is_conjunctive) {
      conjunctive_segments.push_back(&segments[*segment_set.min()]);
      continue;
    }

    if (segment_set.size() == 1) {
      // If we're here the condition is disjunctive, which means that there is
      // at most one condition group (which we are currently on) and since there
      // is only one segment in the condition group we already know it's
      // probability, just return it.
      return segments[*segment_set.min()].Probability();
    }

    // For a group (s1 OR s2 OR ...), compute the union of their definitions.
    std::vector<const Segment*> union_segments;
    for (unsigned s_index : segment_set) {
      const auto& s = segments[s_index];
      union_segments.push_back(&s);
    }

    // TODO(garretrieger): The full probability bound should be utilized here.
    return calculator.ComputeMergedProbability(union_segments).Min();
  }

  return calculator.ComputeConjunctiveProbability(conjunctive_segments).Min();
}

StatusOr<double> ActivationCondition::MergedProbability(
    Span<const Segment> segments, const SegmentSet& merged_segments,
    const Segment& merged_segment,
    const ProbabilityCalculator& calculator) const {
  std::vector<const Segment*> conjunctive_segments;

  bool is_conjunctive = conditions_.size() > 1;
  for (const auto& segment_set : conditions_) {
    if (is_conjunctive && segment_set.size() != 1) {
      // Composite conditions (eg. (a or b) and (c or d)) may have repeated
      // segments in each conjunctive group (eg. (a or b) and (a or d)) which
      // requires special analysis to correctly determine probability. For our
      // current use cases we don't need to support this.
      return absl::UnimplementedError(
          "Calculating probability of composite conditions is not "
          "supported.");
    }

    if (is_conjunctive) {
      segment_index_t s_index = *segment_set.min();
      if (!merged_segments.contains(s_index)) {
        conjunctive_segments.push_back(&segments[s_index]);
      } else {
        conjunctive_segments.push_back(&merged_segment);
      }
      continue;
    }

    if (segment_set.is_subset_of(merged_segments)) {
      // Post merge the segment will be equal to merged_segment, so we can just
      // use it's probability directly.
      return merged_segment.Probability();
    }

    if (segment_set.size() == 1) {
      return segments[*segment_set.min()].Probability();
    }

    // For a group (s1 OR s2 OR ...), compute the union of their definitions.
    bool has_merged = false;
    std::vector<const Segment*> union_segments;
    for (unsigned s_index : segment_set) {
      if (!has_merged && merged_segments.contains(s_index)) {
        has_merged = true;
      }
      union_segments.push_back(&segments[s_index]);
    }

    if (has_merged) {
      // the condition group intersects with the merged set so need to union
      // in all of the merged segments to get the probability.
      union_segments.push_back(&merged_segment);
    }

    return calculator.ComputeMergedProbability(union_segments).Min();
  }

  return calculator.ComputeConjunctiveProbability(conjunctive_segments).Min();
}

}  // namespace ift::encoder