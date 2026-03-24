import tseslint from 'typescript-eslint';

const SRC_TS = ['src/**/*.{ts,tsx}'];

export default [
  {
    ignores: ['dist/**', 'node_modules/**', 'src-tauri/**'],
  },
  {
    files: SRC_TS,
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
      },
    },
    rules: {
      // Global boundary: consume feature/entity public APIs only.
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: ['**/features/*/{model,ui,api,lib}/*'],
              message: 'Import features via public API (features/<slice>/index.ts).',
            },
            {
              group: ['**/entities/*/{model,lib}/*'],
              message: 'Import entities via public API (entities/<slice>/index.ts).',
            },
          ],
        },
      ],
    },
  },
  {
    files: ['src/features/**/*.{ts,tsx}'],
    rules: {
      // Feature slices must be decoupled from other features.
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: ['**/features/*'],
              message:
                'Feature-to-feature imports are forbidden. Use DI or compose via widgets/pages.',
            },
            {
              group: ['**/entities/*/{model,lib}/*'],
              message: 'Import entities through entity public API.',
            },
          ],
        },
      ],
    },
  },
  {
    files: ['src/entities/**/*.{ts,tsx}'],
    rules: {
      // Entities are lower-level and must not depend on upper layers.
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            '**/features/**',
            '**/widgets/**',
            '**/pages/**',
            '**/app/**',
          ],
        },
      ],
    },
  },
  {
    files: ['src/shared/**/*.{ts,tsx}'],
    rules: {
      // Shared stays independent from domain/application layers.
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            '**/entities/**',
            '**/features/**',
            '**/widgets/**',
            '**/pages/**',
            '**/app/**',
          ],
        },
      ],
    },
  },
  {
    files: ['src/widgets/**/*.{ts,tsx}'],
    rules: {
      // Widgets should not depend on pages/app.
      'no-restricted-imports': [
        'error',
        {
          patterns: ['**/pages/**', '**/app/**'],
        },
      ],
    },
  },
  {
    files: ['src/pages/**/*.{ts,tsx}'],
    rules: {
      // Pages should not depend on app layer internals.
      'no-restricted-imports': [
        'error',
        {
          patterns: ['**/app/**'],
        },
      ],
    },
  },
];
