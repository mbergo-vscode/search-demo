import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, (process as any).cwd(), '');
  return {
    plugins: [react()],
    define: {
      // Use the build-time key if available, otherwise use a placeholder for runtime replacement
      'process.env.API_KEY': JSON.stringify(env.API_KEY || "__API_KEY_PLACEHOLDER__")
    },
    server: {
      port: 10000,
      host: true
    },
    preview: {
      port: 10000,
      host: true
    }
  };
});