# Build from the repository root so the shared data/ folder is available:
#   docker build -f apps/web/Dockerfile .
FROM node:22-alpine AS deps
WORKDIR /repo/apps/web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:22-alpine AS build
WORKDIR /repo/apps/web
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /repo/apps/web/node_modules ./node_modules
COPY apps/web ./
# Career pages are generated from data/ at build time.
COPY data /repo/data
# Rewrites are compiled into the build, so the API address must be known here.
ARG API_INTERNAL_URL=http://api:8000
ENV API_INTERNAL_URL=$API_INTERNAL_URL
RUN npm run build

FROM node:22-alpine AS run
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build /repo/apps/web/.next/standalone ./
COPY --from=build /repo/apps/web/.next/static ./.next/static
COPY --from=build /repo/apps/web/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
