FROM node:20-alpine

WORKDIR /app

COPY package*.json ./

# Install dependencies
RUN npm cache clean --force && npm install

COPY . .

# Next.js collects telemetry data about general usage.
# Learn more here: https://nextjs.org/telemetry
# Uncomment the following line in case you want to disable telemetry during the build.
ENV NEXT_TELEMETRY_DISABLED 1

EXPOSE 3000

CMD ["npm", "run", "dev"]
