This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Environment Variables

Copy `.env.example` to `.env.local` and configure your environment variables:

```bash
cp .env.example .env.local
```

Set your Modal API URL:
```
NEXT_PUBLIC_MODAL_API_URL=https://your-modal-api-url.modal.run
```

## Deploy on Cloudflare Pages

This project is configured with [OpenNext](https://opennext.js.org/) for Cloudflare Pages deployment.

### Prerequisites
- [Wrangler CLI](https://developers.cloudflare.com/workers/wrangler/install-and-update/) installed
- Cloudflare account

### Local Preview
```bash
npm run preview
```

### Deploy to Cloudflare Pages
```bash
npm run deploy
```

### Setting Environment Variables in Cloudflare

After deploying, set your environment variables in the Cloudflare dashboard:
1. Go to Workers & Pages > your project > Settings > Environment Variables
2. Add `NEXT_PUBLIC_MODAL_API_URL` with your Modal API URL

Or via Wrangler CLI:
```bash
wrangler pages secret put NEXT_PUBLIC_MODAL_API_URL
```

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
- [OpenNext Cloudflare](https://opennext.js.org/cloudflare) - OpenNext documentation for Cloudflare.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

