/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverActions: { allowedOrigins: ['localhost'] },
  },
  webpack(config) {
    config.module.rules.push({
      test: /\.worker\.ts$/,
      use: { loader: 'worker-loader', options: { inline: 'fallback' } }
    });
    return config;
  }
}
export default nextConfig;
