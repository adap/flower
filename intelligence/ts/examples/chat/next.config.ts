import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Guide on how to configure for huggingface/transformers
  // https://huggingface.co/docs/transformers.js/tutorials/next#step-2-install-and-configure-transformersjs
  // Override the default webpack configuration
  webpack: (config) => {
    // See https://webpack.js.org/configuration/resolve/#resolvealias
    config.resolve.alias = {
      ...config.resolve.alias,
      sharp$: false,
      'onnxruntime-node$': false,
    };
    return config;
  },
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
