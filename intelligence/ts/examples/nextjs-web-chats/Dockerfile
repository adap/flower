# Install dependencies and build Next.js
FROM node:20.19.0 AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Run the app
FROM node:20.19.0
WORKDIR /app
COPY --from=builder /app .
EXPOSE 3000
CMD ["npm", "run", "dev"]
