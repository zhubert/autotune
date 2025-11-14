import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// https://astro.build/config
export default defineConfig({
  site: "https://www.zhubert.com",
  base: import.meta.env.PROD ? "/autotune" : "/",
  trailingSlash: "always",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: "LLM Post-Training",
      description:
        "Learn how to fine-tune language models using SFT, RLHF, and DPO",
      social: [
        {
          icon: "github",
          label: "Github",
          href: "https://github.com/zhubert/autotune",
        },
      ],
      sidebar: [
        {
          label: "Introduction",
          items: [
            { label: "What is Post-Training?", link: "/" },
            { label: "Why Post-Training Matters", link: "/why-post-training/" },
            { label: "Project Overview", link: "/overview/" },
          ],
        },
        {
          label: "Supervised Fine-Tuning (SFT)",
          items: [
            { label: "Introduction to SFT", link: "/sft/" },
            { label: "Instruction Formatting", link: "/sft/formatting/" },
            { label: "Loss Masking", link: "/sft/loss-masking/" },
            { label: "Training Loop", link: "/sft/training/" },
            { label: "LoRA Fine-Tuning", link: "/sft/lora/" },
          ],
        },
        {
          label: "Reward Modeling",
          items: [
            { label: "What are Reward Models?", link: "/reward/" },
            { label: "Preference Data", link: "/reward/preference-data/" },
            { label: "Training Reward Models", link: "/reward/training/" },
            { label: "Evaluation", link: "/reward/evaluation/" },
          ],
        },
        {
          label: "RLHF with PPO",
          items: [
            { label: "Introduction to RLHF", link: "/rlhf/" },
            { label: "PPO Algorithm", link: "/rlhf/ppo/" },
            { label: "KL Divergence Penalty", link: "/rlhf/kl-penalty/" },
            { label: "Training Dynamics", link: "/rlhf/dynamics/" },
            { label: "Reference Models", link: "/rlhf/reference/" },
          ],
        },
        {
          label: "Direct Preference Optimization",
          items: [
            { label: "Introduction to DPO", link: "/dpo/" },
            { label: "DPO vs RLHF", link: "/dpo/vs-rlhf/" },
            { label: "DPO Loss Function", link: "/dpo/loss/" },
            { label: "Training with DPO", link: "/dpo/training/" },
          ],
        },
        {
          label: "Advanced Topics",
          items: [
            { label: "Memory Optimization", link: "/advanced/memory/" },
            { label: "Hyperparameter Tuning", link: "/advanced/hyperparams/" },
            { label: "Evaluation Metrics", link: "/advanced/evaluation/" },
            { label: "Common Pitfalls", link: "/advanced/pitfalls/" },
          ],
        },
        {
          label: "Getting Started",
          items: [{ label: "Try It Yourself", link: "/try-it/" }],
        },
      ],
      customCss: ["./src/styles/custom.css"],
    }),
  ],
});
