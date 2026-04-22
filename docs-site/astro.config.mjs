import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://microresolve.dev',
  integrations: [
    starlight({
      title: 'MicroResolve',
      description: 'Sub-millisecond intent resolution. No embeddings, no GPU, no retraining.',
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
            { label: 'Rust', slug: 'rust/quickstart' },
            { label: 'Python', slug: 'python/quickstart' },
            { label: 'Node.js / TypeScript', slug: 'node/quickstart' },
          ],
        },
        {
          label: 'Concepts',
          items: [
            { label: 'How It Works', slug: 'concepts' },
            { label: 'Benchmarks', slug: 'benchmarks' },
            { label: 'Limitations', slug: 'limitations' },
          ],
        },
        {
          label: 'API Reference',
          items: [
            { label: 'Rust API', slug: 'rust/api' },
            { label: 'Python API', slug: 'python/api' },
            { label: 'Node.js API', slug: 'node/api' },
            { label: 'HTTP Server API', slug: 'server/api' },
          ],
        },
      ],
      customCss: ['./src/styles/custom.css'],
    }),
  ],
});
