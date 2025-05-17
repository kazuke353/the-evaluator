FROM node:18-slim

WORKDIR /app

# Install TypeScript and ts-node for execution
RUN npm install -g typescript ts-node

# Install necessary dependencies for code evaluation
RUN npm init -y && \
    npm install --save-dev @types/node chai mocha @types/chai @types/mocha

# Copy the sandbox script
COPY ts-sandbox.js .

# Set the entrypoint
ENTRYPOINT ["node", "ts-sandbox.js"]
