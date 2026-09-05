import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiUrl = (env.VITE_API_URL || env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

  return {
    plugins: [react()],
    server: {
      port: 3000,
      strictPort: true,
      proxy: {
        '/predict': {
          target: apiUrl,
          changeOrigin: true,
          configure: (proxy) => {
            proxy.on('error', (err, _req, res) => {
              if (res.writeHead) {
                res.writeHead(502, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: `Inference backend unavailable at ${apiUrl}` }));
              }
            });
          },
        },
      },
    },
  };
});
