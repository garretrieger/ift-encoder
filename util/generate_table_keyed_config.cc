#include <google/protobuf/text_format.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common/font_data.h"
#include "common/font_helper.h"
#include "common/int_set.h"
#include "util/load_codepoints.h"
#include "util/segmentation_plan.pb.h"

using common::FontData;
using common::FontHelper;
using common::IntSet;
using google::protobuf::TextFormat;

ABSL_FLAG(
    std::optional<std::string>, font, std::nullopt,
    "Optional, path to a font. If provided the generated config will add an "
    "additional segment if needed that covers any codepoints found in the font "
    "which are not covered by the input subset files.");

template <typename ProtoType>
ProtoType ToSetProto(const IntSet& set) {
  ProtoType values;
  for (uint32_t v : set) {
    values.add_values(v);
  }
  return values;
}

/*
 * This utility takes a font + a list of code point subsets and emits an IFT
 * encoder config that will configure the font to be extended by table keyed
 * patches (where each subset is an extension segment).
 *
 * This config can be appended onto a config which configures the glyph keyed
 * segmentation plan to produce a complete mixed mode configuration.
 *
 * Usage:
 * generate_table_keyed_config <initial font subset fil> <table keyed subset 1
 * file> [... <table keyed subset file n>]
 *
 * Where a subset file lists one codepoint per line in hexadecimal format:
 * 0xXXXX.
 *
 * If you don't want the config to contain an initial codepoint set, pass an
 * empty file as the first argument.
 */

int main(int argc, char** argv) {
  auto args = absl::ParseCommandLine(argc, argv);

  if (args.size() <= 1) {
    std::cerr << "Usage:" << std::endl
              << "generate_table_keyed_config <initial font subset fil> "
                 "<table keyed subset 1 file> [... <table keyed subset file n>]"
              << std::endl
              << std::endl
              << "Where a subset file lists one codepoint per line in "
                 "hexadecimal format: 0xXXXX"
              << std::endl
              << std::endl
              << "If you don't want the config to contain an initial codepoint "
                 "set, pass an empty file as the first argument."
              << std::endl;
    return -1;
  }

  std::vector<IntSet> sets;
  bool first = true;
  for (const char* arg : args) {
    if (first) {
      first = false;
      continue;
    }
    IntSet set;
    auto result = util::LoadCodepointsOrdered(arg);
    if (!result.ok()) {
      std::cerr << "Failed to load codepoints from " << arg << ": "
                << result.status() << std::endl;
      return -1;
    }

    set.insert(result->begin(), result->end());
    sets.push_back(set);
  }

  std::optional<std::string> input_font = absl::GetFlag(FLAGS_font);
  if (input_font.has_value()) {
    // If a font is supplied check if it contains any codepoints not accounted
    // for in an input subset. Add all of these to one last segment.
    auto font_data = util::LoadFile(input_font->c_str());
    if (!font_data.ok()) {
      std::cerr << "Failed to load font, " << *input_font << std::endl;
      return -1;
    }

    auto face = font_data->face();
    auto font_codepoints = FontHelper::ToCodepointsSet(face.get());
    for (const auto& set : sets) {
      for (uint32_t v : set) {
        font_codepoints.erase(v);
      }
    }

    if (!font_codepoints.empty()) {
      sets.push_back(font_codepoints);
    }
  }

  SegmentationPlan config;

  bool initial = true;
  for (const auto& set : sets) {
    if (initial) {
      initial = false;
      if (!set.empty()) {
        *config.mutable_initial_codepoints() = ToSetProto<Codepoints>(set);
      }
      continue;
    }

    if (!set.empty()) {
      *config.add_non_glyph_codepoint_segmentation() =
          ToSetProto<Codepoints>(set);
    }
  }

  std::string config_string;
  TextFormat::PrintToString(config, &config_string);
  std::cout << config_string;

  return 0;
}