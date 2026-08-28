import * as THREE from "three";

const SUPPORT_SIGMA = 3.35;

function instancedAttribute(values, itemSize) {
  return new THREE.InstancedBufferAttribute(values, itemSize, false);
}

function splitColorGradients(flatGradients, count) {
  const rows = [
    new Float32Array(count * 3),
    new Float32Array(count * 3),
    new Float32Array(count * 3),
  ];
  for (let index = 0; index < count; index += 1) {
    const source = index * 9;
    const target = index * 3;
    for (let channel = 0; channel < 3; channel += 1) {
      rows[0][target + channel] = flatGradients[source + channel];
      rows[1][target + channel] = flatGradients[source + 3 + channel];
      rows[2][target + channel] = flatGradients[source + 6 + channel];
    }
  }
  return rows;
}

export class SliceRenderer {
  constructor(field, aspect) {
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-aspect, aspect, 1, -1, -1, 1);
    this.target = new THREE.WebGLRenderTarget(612, 320, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
    });
    this.target.texture.colorSpace = THREE.LinearSRGBColorSpace;

    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0],
        3,
      ),
    );
    geometry.setAttribute("aCenter", instancedAttribute(field.arrays.centers, 3));
    geometry.setAttribute("aVelocity", instancedAttribute(field.arrays.slice_velocities, 2));
    geometry.setAttribute("aSliceRoot", instancedAttribute(field.arrays.slice_roots, 4));
    geometry.setAttribute("aColor", instancedAttribute(field.arrays.colors, 3));
    geometry.setAttribute("aWeight", instancedAttribute(field.arrays.weights, 1));
    geometry.setAttribute("aTimeSigma", instancedAttribute(field.arrays.time_sigmas, 1));
    const gradientRows = splitColorGradients(field.arrays.color_gradients, field.field.count);
    geometry.setAttribute("aGradR", instancedAttribute(gradientRows[0], 3));
    geometry.setAttribute("aGradG", instancedAttribute(gradientRows[1], 3));
    geometry.setAttribute("aGradB", instancedAttribute(gradientRows[2], 3));
    geometry.instanceCount = field.field.count;

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: -field.field.t_scale },
        uAspect: { value: aspect },
        uSupportSigma: { value: SUPPORT_SIGMA },
      },
      vertexShader: `
        uniform float uTime;
        uniform float uAspect;
        uniform float uSupportSigma;
        attribute vec3 aCenter;
        attribute vec2 aVelocity;
        attribute vec4 aSliceRoot;
        attribute vec3 aColor;
        attribute float aWeight;
        attribute float aTimeSigma;
        attribute vec3 aGradR;
        attribute vec3 aGradG;
        attribute vec3 aGradB;
        varying vec2 vStandard;
        varying vec3 vColor;
        varying float vTemporalWeight;

        void main() {
          float dt = uTime - aCenter.z;
          vec2 standard = position.xy * uSupportSigma;
          mat2 root = mat2(
            aSliceRoot.x, aSliceRoot.z,
            aSliceRoot.y, aSliceRoot.w
          );
          vec2 offset = root * standard;
          vec2 center = aCenter.xy + aVelocity * dt;
          vec3 delta = vec3(offset, dt);
          vStandard = standard;
          vColor = aColor + vec3(
            dot(aGradR, delta),
            dot(aGradG, delta),
            dot(aGradB, delta)
          );
          float sigma = max(aTimeSigma, 0.00001);
          vTemporalWeight = aWeight * exp(-0.5 * dt * dt / (sigma * sigma));
          gl_Position = vec4(
            (center.x + offset.x) * uAspect,
            -(center.y + offset.y),
            0.0,
            1.0
          );
        }
      `,
      fragmentShader: `
        varying vec2 vStandard;
        varying vec3 vColor;
        varying float vTemporalWeight;

        void main() {
          float radiusSquared = dot(vStandard, vStandard);
          if (radiusSquared > ${SUPPORT_SIGMA.toFixed(2)} * ${SUPPORT_SIGMA.toFixed(2)}) discard;
          float gaussian = exp(-0.5 * radiusSquared) * vTemporalWeight;
          vec3 contribution = max(vColor, vec3(0.0)) * gaussian;
          if (max(max(contribution.r, contribution.g), contribution.b) < 0.00005) discard;
          gl_FragColor = vec4(contribution, 1.0);
        }
      `,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
    });
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
    this.background = new THREE.Color(...field.field.background);
  }

  render(renderer, time) {
    this.material.uniforms.uTime.value = time;
    const previousTarget = renderer.getRenderTarget();
    const previousColor = renderer.getClearColor(new THREE.Color());
    const previousAlpha = renderer.getClearAlpha();
    renderer.setRenderTarget(this.target);
    renderer.setClearColor(this.background, 1);
    renderer.clear(true, false, false);
    renderer.render(this.scene, this.camera);
    renderer.setRenderTarget(previousTarget);
    renderer.setClearColor(previousColor, previousAlpha);
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.target.dispose();
  }
}
