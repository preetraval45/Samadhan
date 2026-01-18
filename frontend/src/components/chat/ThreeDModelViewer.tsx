'use client'

import { useEffect, useRef, useState } from 'react'
import {
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Download,
  Grid3x3,
  Eye,
} from 'lucide-react'

interface ThreeDModelViewerProps {
  modelUrl: string
  fileName: string
  onClose?: () => void
}

export function ThreeDModelViewer({
  modelUrl,
  fileName,
  onClose,
}: ThreeDModelViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showWireframe, setShowWireframe] = useState(false)
  const [showGrid, setShowGrid] = useState(true)
  const [rotation, setRotation] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)

  // Three.js imports (loaded dynamically)
  const sceneRef = useRef<any>(null)
  const cameraRef = useRef<any>(null)
  const rendererRef = useRef<any>(null)
  const modelRef = useRef<any>(null)
  const animationFrameRef = useRef<number | null>(null)

  useEffect(() => {
    let mounted = true

    const initThreeJS = async () => {
      try {
        // Dynamically import Three.js
        const THREE = await import('three')
        const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js')
        const { OBJLoader } = await import('three/examples/jsm/loaders/OBJLoader.js')
        const { FBXLoader } = await import('three/examples/jsm/loaders/FBXLoader.js')
        const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js')

        if (!mounted || !canvasRef.current) return

        // Scene setup
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0x1a1a1a)
        sceneRef.current = scene

        // Camera
        const camera = new THREE.PerspectiveCamera(
          75,
          canvasRef.current.clientWidth / canvasRef.current.clientHeight,
          0.1,
          1000
        )
        camera.position.z = 5
        cameraRef.current = camera

        // Renderer
        const renderer = new THREE.WebGLRenderer({
          canvas: canvasRef.current,
          antialias: true,
        })
        renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight)
        renderer.setPixelRatio(window.devicePixelRatio)
        rendererRef.current = renderer

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
        scene.add(ambientLight)

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
        directionalLight.position.set(10, 10, 10)
        scene.add(directionalLight)

        // Grid
        if (showGrid) {
          const gridHelper = new THREE.GridHelper(10, 10)
          scene.add(gridHelper)
        }

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.dampingFactor = 0.05

        // Load model based on file extension
        const fileExt = fileName.split('.').pop()?.toLowerCase()
        let loader: any

        if (fileExt === 'gltf' || fileExt === 'glb') {
          loader = new GLTFLoader()
        } else if (fileExt === 'obj') {
          loader = new OBJLoader()
        } else if (fileExt === 'fbx') {
          loader = new FBXLoader()
        } else {
          throw new Error(`Unsupported file format: ${fileExt}`)
        }

        loader.load(
          modelUrl,
          (object: any) => {
            if (!mounted) return

            const model = fileExt === 'gltf' || fileExt === 'glb' ? object.scene : object

            // Center and scale model
            const box = new THREE.Box3().setFromObject(model)
            const center = box.getCenter(new THREE.Vector3())
            const size = box.getSize(new THREE.Vector3())

            const maxDim = Math.max(size.x, size.y, size.z)
            const scale = 4 / maxDim
            model.scale.multiplyScalar(scale)

            model.position.sub(center.multiplyScalar(scale))

            scene.add(model)
            modelRef.current = model

            setIsLoading(false)
          },
          (progress: any) => {
            const percent = (progress.loaded / progress.total) * 100
            console.log(`Loading: ${percent.toFixed(2)}%`)
          },
          (error: any) => {
            console.error('Error loading model:', error)
            setError('Failed to load 3D model')
            setIsLoading(false)
          }
        )

        // Animation loop
        const animate = () => {
          animationFrameRef.current = requestAnimationFrame(animate)

          if (modelRef.current) {
            modelRef.current.rotation.x = rotation.x
            modelRef.current.rotation.y = rotation.y
          }

          controls.update()
          renderer.render(scene, camera)
        }
        animate()

        // Handle resize
        const handleResize = () => {
          if (!canvasRef.current) return

          camera.aspect = canvasRef.current.clientWidth / canvasRef.current.clientHeight
          camera.updateProjectionMatrix()
          renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight)
        }
        window.addEventListener('resize', handleResize)

        return () => {
          window.removeEventListener('resize', handleResize)
        }
      } catch (err: any) {
        console.error('Error initializing Three.js:', err)
        setError(err.message || 'Failed to initialize 3D viewer')
        setIsLoading(false)
      }
    }

    initThreeJS()

    return () => {
      mounted = false
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      if (rendererRef.current) {
        rendererRef.current.dispose()
      }
    }
  }, [modelUrl, fileName, rotation, showGrid])

  const handleResetView = () => {
    setRotation({ x: 0, y: 0 })
    setZoom(1)
    if (cameraRef.current) {
      cameraRef.current.position.set(0, 0, 5)
    }
  }

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev + 0.2, 3))
    if (cameraRef.current) {
      cameraRef.current.position.z -= 0.5
    }
  }

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev - 0.2, 0.5))
    if (cameraRef.current) {
      cameraRef.current.position.z += 0.5
    }
  }

  const toggleWireframe = () => {
    setShowWireframe(!showWireframe)
    if (modelRef.current) {
      modelRef.current.traverse((child: any) => {
        if (child.isMesh) {
          child.material.wireframe = !showWireframe
        }
      })
    }
  }

  return (
    <div className="relative w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ display: isLoading || error ? 'none' : 'block' }}
      />

      {/* Loading State */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <div className="text-center">
            <div className="inline-block w-12 h-12 border-4 border-brand-cyan border-t-transparent rounded-full animate-spin mb-4" />
            <p className="text-white">Loading 3D model...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <div className="text-center text-red-400">
            <p className="text-lg font-medium mb-2">Failed to load model</p>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* Controls */}
      {!isLoading && !error && (
        <div className="absolute top-4 right-4 flex flex-col gap-2">
          <button
            onClick={handleResetView}
            className="p-2 bg-white/10 hover:bg-white/20 backdrop-blur-sm rounded-lg transition-colors"
            title="Reset view"
          >
            <RotateCcw className="w-5 h-5 text-white" />
          </button>

          <button
            onClick={handleZoomIn}
            className="p-2 bg-white/10 hover:bg-white/20 backdrop-blur-sm rounded-lg transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="w-5 h-5 text-white" />
          </button>

          <button
            onClick={handleZoomOut}
            className="p-2 bg-white/10 hover:bg-white/20 backdrop-blur-sm rounded-lg transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="w-5 h-5 text-white" />
          </button>

          <button
            onClick={toggleWireframe}
            className={`p-2 backdrop-blur-sm rounded-lg transition-colors ${
              showWireframe
                ? 'bg-brand-cyan text-white'
                : 'bg-white/10 hover:bg-white/20 text-white'
            }`}
            title="Toggle wireframe"
          >
            <Grid3x3 className="w-5 h-5" />
          </button>

          <button
            onClick={() => setShowGrid(!showGrid)}
            className={`p-2 backdrop-blur-sm rounded-lg transition-colors ${
              showGrid
                ? 'bg-brand-cyan text-white'
                : 'bg-white/10 hover:bg-white/20 text-white'
            }`}
            title="Toggle grid"
          >
            <Eye className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Info Bar */}
      {!isLoading && !error && (
        <div className="absolute bottom-4 left-4 right-4 bg-white/10 backdrop-blur-sm rounded-lg p-3">
          <div className="flex items-center justify-between text-white text-sm">
            <span className="font-medium">{fileName}</span>
            <div className="flex items-center gap-4">
              <span>Zoom: {(zoom * 100).toFixed(0)}%</span>
              <button
                onClick={() => window.open(modelUrl, '_blank')}
                className="flex items-center gap-1 hover:text-brand-cyan transition-colors"
              >
                <Download className="w-4 h-4" />
                Download
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
