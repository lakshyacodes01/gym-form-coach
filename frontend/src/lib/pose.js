// Pure geometry helpers, extracted so they can be unit-tested.
export function angleAt(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y
  const v2x = c.x - b.x, v2y = c.y - b.y
  const dot = v1x * v2x + v1y * v2y
  const mag = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y)
  const cos = Math.max(-1, Math.min(1, dot / mag))
  return (Math.acos(cos) * 180) / Math.PI
}

export function legVisibility(lm, hip, knee, ankle) {
  return ((lm[hip].visibility ?? 0) + (lm[knee].visibility ?? 0) + (lm[ankle].visibility ?? 0)) / 3
}