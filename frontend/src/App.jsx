import { useEffect, useRef, useState } from 'react'
import { PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision'

const DOWN_ANGLE = 100
const UP_ANGLE = 160
const MIN_DROP = 0.5

function angleAt(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y
  const v2x = c.x - b.x, v2y = c.y - b.y
  const dot = v1x * v2x + v1y * v2y
  const mag = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y)
  const cos = Math.max(-1, Math.min(1, dot / mag))
  return (Math.acos(cos) * 180) / Math.PI
}

function legVisibility(lm, hip, knee, ankle) {
  return ((lm[hip].visibility ?? 0) + (lm[knee].visibility ?? 0) + (lm[ankle].visibility ?? 0)) / 3
}

export default function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const repsRef = useRef(0)
  const phaseRef = useRef('up')
  const angleRef = useRef(0)
  const depthRef = useRef(0)
  const standingHipYRef = useRef(null)
  const recordingRef = useRef(false)
  const labelRef = useRef('good')
  const recordedRepsRef = useRef([])
  const currentRepRef = useRef(null)
  const formRef = useRef(null)

  const [status, setStatus] = useState('Loading model...')
  const [reps, setReps] = useState(0)
  const [phase, setPhase] = useState('up')
  const [angle, setAngle] = useState(0)
  const [depth, setDepth] = useState(0)
  const [recording, setRecording] = useState(false)
  const [label, setLabel] = useState('good')
  const [goodCount, setGoodCount] = useState(0)
  const [badCount, setBadCount] = useState(0)
  const [lastForm, setLastForm] = useState(null)

  function reset() {
    repsRef.current = 0; phaseRef.current = 'up'; standingHipYRef.current = null
    setReps(0); setPhase('up')
  }
  function selectLabel(l) { labelRef.current = l; setLabel(l) }
  function toggleRecording() { const n = !recordingRef.current; recordingRef.current = n; setRecording(n) }
  function clearData() { recordedRepsRef.current = []; setGoodCount(0); setBadCount(0) }
  function downloadData() {
    const data = { exercise: 'squat', createdAt: new Date().toISOString(), reps: recordedRepsRef.current }
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = `squat-data-${Date.now()}.json`; a.click()
    URL.revokeObjectURL(url)
  }

  useEffect(() => {
    let cancelled = false, rafId = null, uiTimer = null, landmarker = null, stream = null

    async function start() {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
      )
      landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO', numPoses: 1,
      })
      if (cancelled) return

      setStatus('Starting camera...')
      stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (cancelled) return
      const video = videoRef.current
      video.srcObject = stream
      await video.play()

      setStatus('Tracking')
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      const drawing = new DrawingUtils(ctx)
      let lastTime = -1

      function loop() {
        if (cancelled) return
        if (canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth; canvas.height = video.videoHeight
        }
        if (video.currentTime !== lastTime) {
          lastTime = video.currentTime
          const result = landmarker.detectForVideo(video, performance.now())
          ctx.clearRect(0, 0, canvas.width, canvas.height)

          if (result.landmarks.length > 0) {
            const lm = result.landmarks[0]
            drawing.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF88', lineWidth: 3 })
            drawing.drawLandmarks(lm, { color: '#FF4444', radius: 4 })

            const w = canvas.width, h = canvas.height
            const px = (i) => ({ x: lm[i].x * w, y: lm[i].y * h })
            const leftVis = legVisibility(lm, 23, 25, 27)
            const rightVis = legVisibility(lm, 24, 26, 28)
            const useRight = rightVis >= leftVis
            const hip = useRight ? 24 : 23, knee = useRight ? 26 : 25, ankle = useRight ? 28 : 27
            const bestVis = Math.max(leftVis, rightVis)

            if (bestVis > 0.6) {
              const kneeAngle = angleAt(px(hip), px(knee), px(ankle))
              angleRef.current = kneeAngle
              const shoulderY = ((lm[11].y + lm[12].y) / 2) * h
              const hipY = ((lm[23].y + lm[24].y) / 2) * h
              const torso = Math.abs(hipY - shoulderY) || 1
              if (kneeAngle > UP_ANGLE) standingHipYRef.current = hipY
              const base = standingHipYRef.current
              const drop = base == null ? 0 : (hipY - base) / torso
              depthRef.current = drop

              if (phaseRef.current === 'up' && kneeAngle < DOWN_ANGLE && drop > MIN_DROP) {
                phaseRef.current = 'down'
                currentRepRef.current = []
              } else if (phaseRef.current === 'down' && kneeAngle > UP_ANGLE) {
                phaseRef.current = 'up'
                repsRef.current += 1
                const repFrames = currentRepRef.current
                if (recordingRef.current && repFrames?.length > 0) {
                  recordedRepsRef.current.push({ label: labelRef.current, frames: repFrames })
                }
                // Always send the finished rep to the backend for grading.
                if (repFrames && repFrames.length >= 2) {
                  fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frames: repFrames }),
                  })
                    .then((r) => r.json())
                    .then((d) => { formRef.current = d })
                    .catch(() => { formRef.current = { label: 'error' } })
                }
              }

              // Capture frames during the descent (always, so we can grade every rep).
              if (phaseRef.current === 'down' && currentRepRef.current) {
                currentRepRef.current.push(lm.map((p) => [p.x, p.y, p.z, p.visibility ?? 0]))
              }
            }
          }
        }
        rafId = requestAnimationFrame(loop)
      }
      loop()

      uiTimer = setInterval(() => {
        setReps(repsRef.current); setPhase(phaseRef.current)
        setAngle(Math.round(angleRef.current)); setDepth(Number(depthRef.current.toFixed(2)))
        const recs = recordedRepsRef.current
        setGoodCount(recs.filter((r) => r.label === 'good').length)
        setBadCount(recs.filter((r) => r.label === 'bad').length)
        setLastForm(formRef.current)
      }, 100)
    }

    start().catch((err) => setStatus('Error: ' + err.message))
    return () => {
      cancelled = true
      if (rafId) cancelAnimationFrame(rafId)
      if (uiTimer) clearInterval(uiTimer)
      stream?.getTracks().forEach((t) => t.stop())
      landmarker?.close()
    }
  }, [])

  const labelBtn = (active, color) => ({
    padding: '0.4rem 0.8rem', border: 'none', borderRadius: 6, cursor: 'pointer',
    background: active ? color : '#e5e5e5', color: active ? '#fff' : '#000',
  })
  const formColor = lastForm?.label === 'good' ? '#00AA55' : lastForm?.label === 'bad' ? '#CC3333' : '#888'

  return (
    <main style={{ fontFamily: 'system-ui', maxWidth: 720, margin: '2rem auto', padding: '0 1rem' }}>
      <h1>Gym Form Coach</h1>
      <p>Status: <strong>{status}</strong></p>

      <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', margin: '0.5rem 0', flexWrap: 'wrap' }}>
        <div style={{ fontSize: '2.5rem', fontWeight: 700 }}>{reps} reps</div>
        <div>phase: <strong>{phase}</strong></div>
        <div>knee: <strong>{angle}°</strong></div>
        <div>depth: <strong>{depth}</strong></div>
      </div>

      {lastForm && (
        <div style={{ fontSize: '1.6rem', fontWeight: 700, color: formColor, margin: '0.25rem 0' }}>
          last rep form: {lastForm.label}{lastForm.prob_good != null ? ` (${lastForm.prob_good})` : ''}
        </div>
      )}

      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', margin: '0.75rem 0', flexWrap: 'wrap' }}>
        <span>Label:</span>
        <button onClick={() => selectLabel('good')} style={labelBtn(label === 'good', '#00AA55')}>Good form</button>
        <button onClick={() => selectLabel('bad')} style={labelBtn(label === 'bad', '#CC3333')}>Bad form</button>
        <button onClick={toggleRecording}
                style={{ padding: '0.4rem 0.8rem', border: 'none', borderRadius: 6, cursor: 'pointer',
                         background: recording ? '#CC3333' : '#222', color: '#fff' }}>
          {recording ? '● Stop' : 'Record'}
        </button>
        <span>recorded: <strong>{goodCount} good</strong>, <strong>{badCount} bad</strong></span>
        <button onClick={downloadData} style={{ padding: '0.4rem 0.8rem' }}>Download JSON</button>
        <button onClick={clearData} style={{ padding: '0.4rem 0.8rem' }}>Clear</button>
        <button onClick={reset} style={{ padding: '0.4rem 0.8rem' }}>Reset count</button>
      </div>

      <div style={{ position: 'relative', width: '100%' }}>
        <video ref={videoRef} autoPlay playsInline muted
               style={{ width: '100%', borderRadius: 12, background: '#000', display: 'block' }} />
        <canvas ref={canvasRef}
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
      </div>
    </main>
  )
}