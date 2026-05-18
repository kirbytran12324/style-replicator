import { proxyModalApiRequest } from './proxy';

export const dynamic = 'force-dynamic';

export function GET(request: Request) {
  return proxyModalApiRequest(request, []);
}

export function POST(request: Request) {
  return proxyModalApiRequest(request, []);
}

export function PUT(request: Request) {
  return proxyModalApiRequest(request, []);
}

export function PATCH(request: Request) {
  return proxyModalApiRequest(request, []);
}

export function DELETE(request: Request) {
  return proxyModalApiRequest(request, []);
}

export function OPTIONS(request: Request) {
  return proxyModalApiRequest(request, []);
}
