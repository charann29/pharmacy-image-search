/**
 * util.ts — small shared helpers used across Worker modules.
 */

/**
 * Parse a numeric env var value, falling back when absent/empty/non-finite.
 *
 * @param value    the raw env-var string (may be undefined/empty)
 * @param fallback value returned when parsing fails
 * @param opts.positive when true, require the parsed number to be > 0
 */
export function envNumber(
  value: string | undefined | null,
  fallback: number,
  opts: { positive?: boolean } = {},
): number {
  if (value === undefined || value === null || value === "") return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  if (opts.positive && parsed <= 0) return fallback;
  return parsed;
}

/** Return the input list with falsy values removed and duplicates dropped. */
export function dedupe(items: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of items) {
    if (item && !seen.has(item)) {
      seen.add(item);
      out.push(item);
    }
  }
  return out;
}
