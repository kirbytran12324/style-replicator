import { proxyModalApiRequest } from '../proxy';

export const dynamic = 'force-dynamic';

type RouteContext = {
  params: Promise<{ path?: string[] }>;
};

async function getPathSegments(context: RouteContext) {
  const params = await context.params;
  return params.path ?? [];
}

export async function GET(request: Request, context: RouteContext) {
  return proxyModalApiRequest(request, await getPathSegments(context));
}

export async function POST(request: Request, context: RouteContext) {
  return proxyModalApiRequest(request, await getPathSegments(context));
}

export async function PUT(request: Request, context: RouteContext) {
  return proxyModalApiRequest(request, await getPathSegments(context));
}

export async function PATCH(request: Request, context: RouteContext) {
  return proxyModalApiRequest(request, await getPathSegments(context));
}

export async function DELETE(request: Request, context: RouteContext) {
  return proxyModalApiRequest(request, await getPathSegments(context));
}

export async function OPTIONS(request: Request, context: RouteContext) {
  return proxyModalApiRequest(request, await getPathSegments(context));
}
