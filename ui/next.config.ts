import type { NextConfig } from 'next';
import { initOpenNextCloudflareForDev } from '@opennextjs/cloudflare';

if (process.env.NODE_ENV === "development") {
  initOpenNextCloudflareForDev();
}


const nextConfig: NextConfig = {
  devIndicators: {
    buildActivity: false,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '100mb',
    },
  },
  // Required for OpenNext/Cloudflare compatibility
  images: {
    unoptimized: true,
  },
};

export default nextConfig;