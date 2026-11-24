import { type NextRequest, NextResponse } from "next/server";

// Set by the API next to the httpOnly tokens. It holds no secret; it only says a session exists.
const SESSION_HINT = "pathify_session";

const APP_ROUTES = ["/home", "/plan", "/counsellor", "/ksarth", "/settings", "/onboarding", "/assessment", "/results", "/tryouts", "/report"];
const GUEST_ROUTES = ["/signin", "/signup"];

export function middleware(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  const signedIn = request.cookies.get(SESSION_HINT)?.value === "1";

  if (!signedIn && APP_ROUTES.some((route) => pathname === route || pathname.startsWith(`${route}/`))) {
    const url = request.nextUrl.clone();
    url.pathname = "/signin";
    url.search = `?next=${encodeURIComponent(pathname + search)}`;
    return NextResponse.redirect(url);
  }
  if (signedIn && GUEST_ROUTES.includes(pathname) && !request.nextUrl.searchParams.has("error")) {
    const url = request.nextUrl.clone();
    url.pathname = "/home";
    url.search = "";
    return NextResponse.redirect(url);
  }
  return NextResponse.next();
}

export const config = {
  matcher: [
    "/home/:path*",
    "/plan/:path*",
    "/counsellor/:path*",
    "/ksarth/:path*",
    "/settings/:path*",
    "/onboarding/:path*",
    "/assessment/:path*",
    "/results/:path*",
    "/tryouts/:path*",
    "/report/:path*",
    "/signin",
    "/signup",
  ],
};
