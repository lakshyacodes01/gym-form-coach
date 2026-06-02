import { describe, it, expect } from 'vitest'
import { angleAt } from './pose.js'

describe('angleAt', () => {
  it('is 90° for a right angle', () => {
    expect(angleAt({ x: 0, y: 1 }, { x: 0, y: 0 }, { x: 1, y: 0 })).toBeCloseTo(90)
  })
  it('is 180° for a straight leg', () => {
    expect(angleAt({ x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 })).toBeCloseTo(180)
  })
})