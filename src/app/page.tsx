"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState("ยังไม่เริ่ม");
  const [emotion, setEmotion] = useState("-");
  const [conf, setConf] = useState(0);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  async function loadOpenCV() {
    if (typeof window === "undefined") return;

    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;

      script.onload = () => {
        const cv = (window as any).cv;
        if (!cv) return reject(new Error("OpenCV โหลดแล้วแต่ window.cv ไม่มีค่า"));

        const waitReady = () => {
          if ((window as any).cv?.Mat) {
            cvRef.current = (window as any).cv;
            resolve();
          } else {
            setTimeout(waitReady, 50);
          }
        };

        if ("onRuntimeInitialized" in cv) {
          cv.onRuntimeInitialized = () => waitReady();
        } else {
          waitReady();
        }
      };

      script.onerror = () =>
        reject(new Error("โหลด /opencv/opencv.js ไม่สำเร็จ"));

      document.body.appendChild(script);
    });
  }

  async function loadCascade() {
    const cv = cvRef.current;
    if (!cv) throw new Error("cv ยังไม่พร้อม");

    const res = await fetch("/opencv/haarcascade_frontalface_default.xml");
    if (!res.ok) throw new Error("โหลด cascade ไม่สำเร็จ");

    const data = new Uint8Array(await res.arrayBuffer());
    const path = "haarcascade_frontalface_default.xml";

    try {
      cv.FS_unlink(path);
    } catch {}

    cv.FS_createDataFile("/", path, data, true, false, false);

    const faceCascade = new cv.CascadeClassifier();
    if (!faceCascade.load(path)) {
      throw new Error("cascade load() ไม่สำเร็จ");
    }

    faceCascadeRef.current = faceCascade;
  }

  async function loadModel() {
    sessionRef.current = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      { executionProviders: ["wasm"] }
    );

    const res = await fetch("/models/classes.json");
    classesRef.current = await res.json();
  }

  async function startCamera() {
    setStatus("ขอสิทธิ์กล้อง...");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    if (!videoRef.current) return;
    videoRef.current.srcObject = stream;
    await videoRef.current.play();

    setStatus("กำลังทำงาน...");
    requestAnimationFrame(loop);
  }

  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;

    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);

    const img = ctx.getImageData(0, 0, size, size).data;
    const data = new Float32Array(1 * 3 * size * size);

    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        const r = img[i * 4] / 255;
        const g = img[i * 4 + 1] / 255;
        const b = img[i * 4 + 2] / 255;
        data[idx++] = c === 0 ? r : c === 1 ? g : b;
      }
    }

    return new ort.Tensor("float32", data, [1, 3, size, size]);
  }

  function softmax(arr: Float32Array) {
    const max = Math.max(...arr);
    const exps = Array.from(arr).map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
  }

  async function loop() {
    const cv = cvRef.current;
    const faceCascade = faceCascadeRef.current;
    const session = sessionRef.current;
    const classes = classesRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
      requestAnimationFrame(loop);
      return;
    }

    const ctx = canvas.getContext("2d")!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const faces = new cv.RectVector();
    faceCascade.detectMultiScale(gray, faces);

    let best: any = null;
    let bestArea = 0;

    for (let i = 0; i < faces.size(); i++) {
      const r = faces.get(i);
      const area = r.width * r.height;
      if (area > bestArea) {
        bestArea = area;
        best = r;
      }
      ctx.strokeStyle = "lime";
      ctx.lineWidth = 2;
      ctx.strokeRect(r.x, r.y, r.width, r.height);
    }

    if (best) {
      const fc = document.createElement("canvas");
      fc.width = best.width;
      fc.height = best.height;
      fc.getContext("2d")!.drawImage(
        canvas,
        best.x,
        best.y,
        best.width,
        best.height,
        0,
        0,
        best.width,
        best.height
      );

      const input = preprocessToTensor(fc);
      const feeds: any = {};
      feeds[session.inputNames[0]] = input;

      const out = await session.run(feeds);
      const logits = out[session.outputNames[0]].data as Float32Array;
      const probs = softmax(logits);

      const idx = probs.indexOf(Math.max(...probs));
      setEmotion(classes[idx]);
      setConf(probs[idx]);

      ctx.fillStyle = "black";
      ctx.fillRect(best.x, best.y - 24, 220, 24);
      ctx.fillStyle = "white";
      ctx.fillText(
        `${classes[idx]} ${(probs[idx] * 100).toFixed(1)}%`,
        best.x + 6,
        best.y - 6
      );
    }

    src.delete();
    gray.delete();
    faces.delete();

    requestAnimationFrame(loop);
  }

  useEffect(() => {
    (async () => {
      try {
        setStatus("กำลังโหลด OpenCV...");
        await loadOpenCV();

        setStatus("กำลังโหลด Cascade...");
        await loadCascade();

        setStatus("กำลังโหลดโมเดล...");
        await loadModel();

        setStatus("พร้อม กด Start");
      } catch (e: any) {
        setStatus(e.message);
      }
    })();
  }, []);

  return (
    <main className="min-h-screen p-6 space-y-4">
      <h1 className="text-2xl font-bold">Face Emotion Detection</h1>

      <div>
        สถานะ: {status} <br />
        Emotion: <b>{emotion}</b> ({(conf * 100).toFixed(1)}%)
      </div>

      <button
        onClick={startCamera}
        className="px-4 py-2 bg-black text-white rounded"
      >
        Start Camera
      </button>

      <video ref={videoRef} className="hidden" playsInline />
      <canvas ref={canvasRef} className="w-full max-w-3xl border rounded" />
    </main>
  );
}