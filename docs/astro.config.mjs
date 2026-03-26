import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://spcshft.com",
  integrations: [
    starlight({
      title: "spaceshift",
      description:
        "An open research toolkit powered by LLMs. Branch, navigate, evaluate.",
      customCss: ["./src/styles/custom.css"],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/sdeep27/spaceshift",
        },
      ],
      sidebar: [
        {
          label: "Getting Started",
          items: [
            { label: "Installation", slug: "getting-started/installation" },
            { label: "Quickstart", slug: "getting-started/quickstart" },
          ],
        },
        {
          label: "Core: The LLM Class",
          items: [
            { label: "Overview", slug: "core/overview" },
            { label: "Conversation", slug: "core/conversation" },
            { label: "JSON & Structured Output", slug: "core/json-output" },
            {
              label: "Templates & Batch Processing",
              slug: "core/templates-and-batch",
            },
            { label: "Tool Calling", slug: "core/tool-calling" },
            { label: "Multimodal", slug: "core/multimodal" },
            { label: "Model Selection", slug: "core/model-selection" },
            { label: "Cost Tracking", slug: "core/cost-tracking" },
            {
              label: "History & Persistence",
              slug: "core/history-persistence",
            },
          ],
        },
        {
          label: "Research",
          items: [
            { label: "Overview", slug: "research/overview" },
            {
              label: "Directional Exploration",
              slug: "research/directional-exploration",
            },
            { label: "Research Tree", slug: "research/research-tree" },
            { label: "Research Expand", slug: "research/research-expand" },
            { label: "Prompt Transforms", slug: "research/prompt-transforms" },
            {
              label: "Language Transform",
              slug: "research/language-transform",
            },
            { label: "Prompt Probe", slug: "research/prompt-probe" },
          ],
        },
        {
          label: "Evaluation",
          items: [
            {
              label: "Pairwise Evaluate",
              slug: "evaluation/pairwise-evaluate",
            },
            { label: "Compare Models", slug: "evaluation/compare-models" },
            { label: "Grid Search", slug: "evaluation/grid-search" },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "Model Rankings", slug: "reference/model-rankings" },
            {
              label: "Transforms Catalog",
              slug: "reference/transforms-catalog",
            },
            { label: "Markdown Output", slug: "reference/markdown-output" },
            { label: "Viewer", slug: "reference/viewer" },
          ],
        },
      ],
    }),
  ],
});
