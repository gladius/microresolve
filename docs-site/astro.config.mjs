import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://gladius.github.io',
  base: '/microresolve',
  integrations: [
    starlight({
      title: 'MicroResolve',
      description: 'Pre-LLM decision layer: intent routing, PII detection, safety filtering, tool prefiltering. ~50μs per call, CPU-only, continuous learning.',
      logo: {
        light: './src/assets/logo-light.svg',
        dark: './src/assets/logo-dark.svg',
        replacesTitle: false,
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/gladius/microresolve' },
      ],
      editLink: {
        baseUrl: 'https://github.com/gladius/microresolve/edit/main/docs-site/',
      },
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Introduction', slug: 'index' },
            { label: 'Concepts', slug: 'concepts' },
          ],
        },
        {
          label: 'Rust',
          items: [
            { label: 'Quickstart', slug: 'rust/quickstart' },
            { label: 'API Reference', slug: 'rust/api' },
            { label: 'Connect to Server', slug: 'rust/connect' },
          ],
        },
        {
          label: 'Python',
          items: [
            { label: 'Quickstart', slug: 'python/quickstart' },
            { label: 'API Reference', slug: 'python/api' },
            { label: 'Connect to Server', slug: 'python/connect' },
          ],
        },
        {
          label: 'Node.js',
          items: [
            { label: 'Quickstart', slug: 'node/quickstart' },
            { label: 'API Reference', slug: 'node/api' },
            { label: 'Connect to Server', slug: 'node/connect' },
          ],
        },
        {
          label: 'Server',
          items: [
            { label: 'Install', slug: 'server/install' },
            { label: 'Studio UI', slug: 'server/studio' },
            { label: 'API Reference', slug: 'server/api' },
            { label: 'Git Data Layer', slug: 'server/git-data' },
            { label: 'Auth', slug: 'server/auth' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'Benchmarks', slug: 'benchmarks' },
            { label: 'Threshold Tuning', slug: 'threshold-tuning' },
            { label: 'Limitations', slug: 'limitations' },
          ],
        },
      ],
      customCss: ['./src/styles/custom.css'],
    }),
  ],
});
