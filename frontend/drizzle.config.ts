import { type Config } from "drizzle-kit";

import { env } from "targon/env";

export default {
  schema: "./src/server/db/schema.ts",
  dialect: "postgresql",
  dbCredentials: {
    url: env.DATABASE_URL,
  },
  tablesFilter: ["frontend_*"],
} satisfies Config;
