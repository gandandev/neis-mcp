import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

// Define our MCP agent with NEIS tools
export class MyMCP extends McpAgent {
  server = new McpServer({
    name: "NEIS MCP",
    version: "1.0.0",
  });

  async init() {
    // Helper to call NEIS Open API
    const callNeis = async (
      env: Env,
      endpoint: string,
      params: Record<string, string | number>
    ) => {
      const { NEIS_API_KEY } =
        (env as unknown as { NEIS_API_KEY?: string }) || {};
      if (!NEIS_API_KEY) {
        return {
          content: [
            { type: "text", text: "Missing NEIS_API_KEY in environment." },
          ],
        } as any;
      }

      const base = "https://open.neis.go.kr/hub/";
      const url = new URL(endpoint, base);

      // Always request JSON
      url.searchParams.set("Type", "json");
      // Auth key
      url.searchParams.set("KEY", NEIS_API_KEY);

      for (const [k, v] of Object.entries(params || {})) {
        url.searchParams.set(k, String(v));
      }

      const res = await fetch(url.toString());
      if (!res.ok) {
        return {
          content: [
            {
              type: "text",
              text: `NEIS request failed: ${res.status} ${res.statusText}`,
            },
          ],
        } as any;
      }

      const data = await res.json().catch(() => undefined);
      if (!data) {
        return {
          content: [
            { type: "text", text: "Failed to parse NEIS response as JSON." },
          ],
        } as any;
      }

      // Return raw JSON as string for MCP clients
      return {
        content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
      } as any;
    };

    // Convenience: search schools by name
    this.server.registerTool(
      "Search School",
      {
        description: "Find schools by name (and optional office code)",
        inputSchema: {
          name: z.string().describe("School name (partial allowed)"),
          officeCode: z
            .string()
            .optional()
            .describe("Education office code (ATPT_OFCDC_SC_CODE)"),
          page: z.number().int().positive().default(1),
          size: z.number().int().positive().max(1000).default(100),
        },
      },
      async ({ name, officeCode, page, size }, extra) => {
        const params: Record<string, string | number> = {
          SCHUL_NM: name,
          pIndex: page,
          pSize: size,
        };
        if (officeCode) params.ATPT_OFCDC_SC_CODE = officeCode;
        return callNeis(
          (this as unknown as { env: Env }).env,
          "schoolInfo",
          params
        );
      }
    );

    // Convenience: meals by date or range
    this.server.registerTool(
      "Get Meal Info",
      {
        description: "Get meal info for a date or range",
        inputSchema: {
          officeCode: z.string().describe("ATPT_OFCDC_SC_CODE"),
          schoolCode: z.string().describe("SD_SCHUL_CODE"),
          ymd: z
            .string()
            .regex(/^\d{8}$/)
            .optional()
            .describe("Single date YYYYMMDD"),
          fromYmd: z
            .string()
            .regex(/^\d{8}$/)
            .optional(),
          toYmd: z
            .string()
            .regex(/^\d{8}$/)
            .optional(),
          page: z.number().int().positive().default(1),
          size: z.number().int().positive().max(1000).default(100),
        },
      },
      async (
        { officeCode, schoolCode, ymd, fromYmd, toYmd, page, size },
        extra
      ) => {
        const params: Record<string, string | number> = {
          ATPT_OFCDC_SC_CODE: officeCode,
          SD_SCHUL_CODE: schoolCode,
          pIndex: page,
          pSize: size,
        };
        if (ymd) params.MLSV_YMD = ymd;
        if (fromYmd) params.MLSV_FROM_YMD = fromYmd;
        if (toYmd) params.MLSV_TO_YMD = toYmd;
        return callNeis(
          (this as unknown as { env: Env }).env,
          "mealServiceDietInfo",
          params
        );
      }
    );

    // Convenience: timetable
    this.server.registerTool(
      "Get Timetable",
      {
        description: "Get class timetable for a school and date",
        inputSchema: {
          officeCode: z.string().describe("ATPT_OFCDC_SC_CODE"),
          schoolCode: z.string().describe("SD_SCHUL_CODE"),
          schoolType: z
            .enum(["elementary", "middle", "high", "special"]) // maps to els/mis/his/sps
            .describe("School level"),
          grade: z.string().describe("GRADE (학년)"),
          className: z.string().describe("CLRM_NM (반)"),
          ymd: z
            .string()
            .regex(/^\d{8}$/)
            .describe("ALL_TI_YMD (YYYYMMDD)"),
          page: z.number().int().positive().default(1),
          size: z.number().int().positive().max(1000).default(100),
        },
      },
      async (
        {
          officeCode,
          schoolCode,
          schoolType,
          grade,
          className,
          ymd,
          page,
          size,
        },
        extra
      ) => {
        const endpointMap: Record<string, string> = {
          elementary: "elsTimetable",
          middle: "misTimetable",
          high: "hisTimetable",
          special: "spsTimetable",
        };
        const endpoint = endpointMap[schoolType];
        const params: Record<string, string | number> = {
          ATPT_OFCDC_SC_CODE: officeCode,
          SD_SCHUL_CODE: schoolCode,
          GRADE: grade,
          CLRM_NM: className,
          ALL_TI_YMD: ymd,
          pIndex: page,
          pSize: size,
        };
        return callNeis(
          (this as unknown as { env: Env }).env,
          endpoint,
          params
        );
      }
    );

    // Convenience: school schedule (holidays/events)
    this.server.registerTool(
      "Get Schedule",
      {
        description: "Get school schedule/holidays for a date or range",
        inputSchema: {
          officeCode: z.string().describe("ATPT_OFCDC_SC_CODE"),
          schoolCode: z.string().describe("SD_SCHUL_CODE"),
          ymd: z
            .string()
            .regex(/^\d{8}$/)
            .optional(),
          fromYmd: z
            .string()
            .regex(/^\d{8}$/)
            .optional(),
          toYmd: z
            .string()
            .regex(/^\d{8}$/)
            .optional(),
          page: z.number().int().positive().default(1),
          size: z.number().int().positive().max(1000).default(100),
        },
      },
      async (
        { officeCode, schoolCode, ymd, fromYmd, toYmd, page, size },
        extra
      ) => {
        const params: Record<string, string | number> = {
          ATPT_OFCDC_SC_CODE: officeCode,
          SD_SCHUL_CODE: schoolCode,
          pIndex: page,
          pSize: size,
        };
        if (ymd) params.AA_YMD = ymd;
        if (fromYmd) params.AA_FROM_YMD = fromYmd;
        if (toYmd) params.AA_TO_YMD = toYmd;
        return callNeis(
          (this as unknown as { env: Env }).env,
          "SchoolSchedule",
          params
        );
      }
    );
  }
}

export default {
  fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const url = new URL(request.url);

    if (url.pathname === "/sse" || url.pathname === "/sse/message") {
      return MyMCP.serveSSE("/sse").fetch(request, env, ctx);
    }

    if (url.pathname === "/mcp") {
      return MyMCP.serve("/mcp").fetch(request, env, ctx);
    }

    // Serve favicon if available via ASSETS binding (no moving files)
    if (url.pathname === "/favicon.svg") {
      try {
        const assets = (
          env as unknown as {
            ASSETS?: { fetch?: (r: Request) => Promise<Response> };
          }
        ).ASSETS;
        if (assets && typeof assets.fetch === "function") {
          // attempt to fetch static asset
          return assets.fetch(request);
        }
      } catch {
        // ignore and fall through
      }
      return new Response("", { status: 404 });
    }

    return new Response("Not found", { status: 404 });
  },
};
