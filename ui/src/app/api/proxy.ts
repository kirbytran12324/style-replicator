const BODYLESS_METHODS = new Set(['GET', 'HEAD']);

const getModalApiBaseURL = () => (process.env.NEXT_PUBLIC_MODAL_API_URL || '').replace(/\/$/, '');

const buildTargetURL = (request: Request, pathSegments: string[]) => {
  const baseURL = getModalApiBaseURL();
  if (!baseURL) return null;

  const incomingURL = new URL(request.url);
  const encodedPath = pathSegments.map(segment => encodeURIComponent(segment)).join('/');
  const apiPath = encodedPath ? `/api/${encodedPath}` : '/api';
  return `${baseURL}${apiPath}${incomingURL.search}`;
};

const buildProxyHeaders = (request: Request) => {
  const headers = new Headers(request.headers);
  headers.delete('host');
  headers.delete('content-length');
  return headers;
};

export async function proxyModalApiRequest(request: Request, pathSegments: string[]) {
  const targetURL = buildTargetURL(request, pathSegments);
  if (!targetURL) {
    return Response.json({ detail: 'NEXT_PUBLIC_MODAL_API_URL is not configured' }, { status: 500 });
  }

  const upstream = await fetch(targetURL, {
    method: request.method,
    headers: buildProxyHeaders(request),
    body: BODYLESS_METHODS.has(request.method) ? undefined : await request.arrayBuffer(),
    redirect: 'manual',
  });

  const responseHeaders = new Headers(upstream.headers);
  responseHeaders.delete('content-encoding');
  responseHeaders.delete('content-length');

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders,
  });
}
