import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CREAM = 0xf4f0e6;
const INK = 0x333835;
const TEAL = 0x087f6d;
const SHELL_LIMIT = 500;
const TIME_DEPTH = 1.35;

function makePointCloud(field, aspect) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(field.arrays.centers, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(field.arrays.colors, 3));
  geometry.setAttribute("aTimeSigma", new THREE.BufferAttribute(field.arrays.time_sigmas, 1));
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: -field.field.t_scale },
      uAspect: { value: aspect },
      uTimeDepth: { value: TIME_DEPTH },
    },
    vertexShader: `
      uniform float uTime;
      uniform float uAspect;
      uniform float uTimeDepth;
      attribute vec3 aColor;
      attribute float aTimeSigma;
      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        float sigma = max(aTimeSigma, 0.00001);
        float sliceGain = exp(-0.5 * pow((uTime - position.z) / sigma, 2.0));
        vec3 world = vec3(position.x * uAspect, -position.y, position.z * uTimeDepth);
        vec4 view = modelViewMatrix * vec4(world, 1.0);
        gl_Position = projectionMatrix * view;
        gl_PointSize = clamp((1.2 + 3.8 * sliceGain) * (150.0 / -view.z), 1.0, 9.0);
        vColor = mix(vec3(0.25, 0.28, 0.27), clamp(aColor, 0.0, 1.0), 0.25 + 0.75 * sliceGain);
        vAlpha = 0.025 + 0.64 * sliceGain;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        vec2 delta = gl_PointCoord - vec2(0.5);
        float radius = length(delta);
        if (radius > 0.5) discard;
        float feather = smoothstep(0.5, 0.18, radius);
        gl_FragColor = vec4(vColor, vAlpha * feather);
      }
    `,
    transparent: true,
    depthWrite: false,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  return points;
}

function makeShells(field, aspect) {
  const count = Math.min(SHELL_LIMIT, field.field.count);
  const ranked = Array.from({ length: field.field.count }, (_, index) => index)
    .sort((left, right) => field.arrays.importance[right] - field.arrays.importance[left])
    .slice(0, count);
  const geometry = new THREE.IcosahedronGeometry(1, 1);
  const material = new THREE.MeshBasicMaterial({
    color: 0x2463a7,
    wireframe: true,
    transparent: true,
    opacity: 0.095,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const shells = new THREE.InstancedMesh(geometry, material, count);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  const local = new THREE.Matrix4();
  const display = new THREE.Matrix4().makeScale(aspect, -1, TIME_DEPTH);
  for (let displayIndex = 0; displayIndex < count; displayIndex += 1) {
    const sourceIndex = ranked[displayIndex];
    const center = sourceIndex * 3;
    const rotation = sourceIndex * 4;
    position.fromArray(field.arrays.centers, center);
    quaternion.fromArray(field.arrays.quaternions, rotation);
    scale.fromArray(field.arrays.scales, center).multiplyScalar(2.0);
    local.compose(position, quaternion, scale);
    shells.setMatrixAt(displayIndex, display.clone().multiply(local));
  }
  shells.instanceMatrix.needsUpdate = true;
  shells.frustumCulled = false;
  return shells;
}

function makeBounds(aspect) {
  const geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(2 * aspect, 2, 2 * TIME_DEPTH));
  const material = new THREE.LineBasicMaterial({ color: INK, transparent: true, opacity: 0.28 });
  return new THREE.LineSegments(geometry, material);
}

function makePlane(texture, aspect) {
  const group = new THREE.Group();
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(2 * aspect, 2),
    new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }),
  );
  plane.renderOrder = 2;
  group.add(plane);
  const borderPoints = [
    new THREE.Vector3(-aspect, -1, 0.006),
    new THREE.Vector3(aspect, -1, 0.006),
    new THREE.Vector3(aspect, 1, 0.006),
    new THREE.Vector3(-aspect, 1, 0.006),
    new THREE.Vector3(-aspect, -1, 0.006),
  ];
  const border = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(borderPoints),
    new THREE.LineBasicMaterial({ color: TEAL, linewidth: 2 }),
  );
  border.renderOrder = 3;
  group.add(border);
  return group;
}

export class VolumeScene {
  constructor(canvas, field, sliceTexture) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setClearColor(CREAM, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(CREAM);
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    this.camera.position.set(4.3, 2.55, 4.7);
    this.camera.lookAt(0, 0, 0);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.minDistance = 2.4;
    this.controls.maxDistance = 11;

    this.points = makePointCloud(field, field.field.shape[2] / field.field.shape[1]);
    this.shells = makeShells(field, field.field.shape[2] / field.field.shape[1]);
    this.bounds = makeBounds(field.field.shape[2] / field.field.shape[1]);
    this.playhead = makePlane(sliceTexture, field.field.shape[2] / field.field.shape[1]);
    this.scene.add(this.points, this.shells, this.bounds, this.playhead);
  }

  setTime(time) {
    this.points.material.uniforms.uTime.value = time;
    this.playhead.position.z = time * TIME_DEPTH;
  }

  resize(width, height) {
    const safeWidth = Math.max(1, Math.floor(width));
    const safeHeight = Math.max(1, Math.floor(height));
    this.renderer.setSize(safeWidth, safeHeight, false);
    this.camera.aspect = safeWidth / safeHeight;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
