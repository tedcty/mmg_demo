<script setup lang="ts">
import { ref, onMounted } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Paths
const anthro_path = ref("C:\\Users\\rrag962\\Documents\\shapemodel\\model\\Shoulder 3\\Segmentations\\SSM\\PLSR\\anthro_data.csv");
const ssm_path = ref("C:\\Users\\rrag962\\Documents\\shapemodel\\model\\Shoulder 3\\Segmentations\\SSM\\Combined\\shape_model");
const out_path = ref("C:\\Users\\rrag962\\Documents\\shapemodel\\model\\Shoulder 3\\Segmentations\\EOS\\predicted_model.ply");

// Patient Data
const sex = ref("0");
const age = ref("63");
const height = ref("154.6");
const weight = ref("74.7");
const r_clav_len = ref("170");
const r_hum_len = ref("296");
const r_hum_epi_width = ref("55");

const statusMessage = ref("");
const statusColor = ref("#ffffff");
const isPredicting = ref(false);
const isSavingReport = ref(false);
const isSettingsVisible = ref(false);
const isKinematicVisible = ref(false);

// Scapulothoracic Joint State (4-DOF)
// Default values from literature (Protraction/Abduction, Elevation, Upward Rot, Tilt/Winging)
const r_st_coords = ref({ abduction: 35.0, elevation: 5.0, upward: 15.0, winging: 0.0 });
const l_st_coords = ref({ abduction: 35.0, elevation: 5.0, upward: 15.0, winging: 0.0 });

// Mesh References for dynamic updates
const thoraxMesh = ref<THREE.Mesh | null>(null);
const clavicleMeshes = { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null };
const scapulaMeshes = { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null };
const humerusMeshes = { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null };

// Original Joint Definitions from Python
const jointPivots = { 
    right: { sc: new THREE.Vector3(), ac: new THREE.Vector3(), gh: new THREE.Vector3() },
    left:  { sc: new THREE.Vector3(), ac: new THREE.Vector3(), gh: new THREE.Vector3() } 
};

// Visual Spheres for Joint Centers (Parent vs Child Diagnostics)
const jointMarkersP = {
    right: { sc: null as THREE.Mesh | null, ac: null as THREE.Mesh | null, gh: null as THREE.Mesh | null },
    left:  { sc: null as THREE.Mesh | null, ac: null as THREE.Mesh | null, gh: null as THREE.Mesh | null }
};
const jointMarkersC = {
    right: { sc: null as THREE.Mesh | null, ac: null as THREE.Mesh | null, gh: null as THREE.Mesh | null },
    left:  { sc: null as THREE.Mesh | null, ac: null as THREE.Mesh | null, gh: null as THREE.Mesh | null }
};

// Visual Sprites for Coordinate Labels
const jointLabels = {
    right: { sc: null as THREE.Sprite | null, ac: null as THREE.Sprite | null, gh: null as THREE.Sprite | null },
    left:  { sc: null as THREE.Sprite | null, ac: null as THREE.Sprite | null, gh: null as THREE.Sprite | null }
};

// Local Frame Origin Markers (at 0,0,0 for each bone)
const originMarkers = {
    thorax: null as THREE.Mesh | null,
    clavicle: { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null },
    scapula: { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null },
    humerus: { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null }
};
const originLabels = {
    thorax: null as THREE.Sprite | null,
    clavicle: { right: null as THREE.Sprite | null, left: null as THREE.Sprite | null },
    scapula: { right: null as THREE.Sprite | null, left: null as THREE.Sprite | null },
    humerus: { right: null as THREE.Sprite | null, left: null as THREE.Sprite | null }
};

const initialQuats = {
  clavicle: { right: new THREE.Quaternion(), left: new THREE.Quaternion() },
  scapula: { right: new THREE.Quaternion(), left: new THREE.Quaternion() },
  humerus: { right: new THREE.Quaternion(), left: new THREE.Quaternion() }
};
const initialPositions = {
  clavicle: { right: new THREE.Vector3(), left: new THREE.Vector3() },
  scapula: { right: new THREE.Vector3(), left: new THREE.Vector3() },
  humerus: { right: new THREE.Vector3(), left: new THREE.Vector3() }
};

const viewerContainer = ref<HTMLElement | null>(null);

onMounted(async () => {
  // Listen for Python stdout chunks
  listen("progress-status", (event) => {
    const text = event.payload as string;
    if (text.startsWith("STATUS|")) {
      statusMessage.value = text.replace("STATUS|", "");
      statusColor.value = "#00d1b2";
    } else if (text.startsWith("SUCCESS|")) {
      statusMessage.value = text.replace("SUCCESS|", "");
      statusColor.value = "#48c774";
    } else if (text.startsWith("ERROR|")) {
      statusMessage.value = text.replace("ERROR|", "");
      statusColor.value = "#f14668";
    } else {
      statusMessage.value = text;
    }
  });

  // Initialize Three.js natively
  if (viewerContainer.value) {
    const width = viewerContainer.value.clientWidth;
    const height = viewerContainer.value.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 5000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    viewerContainer.value.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(500, 1000, 500);
    scene.add(dirLight);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    try {
      const response = await fetch('/bones.json');
      const data = await response.json();

      const center = data.center;
      const spread = data.spread;
      
      const sceneCenter = new THREE.Vector3(center[0], center[1], center[2]);
      controls.target.copy(sceneCenter);
      camera.position.set(sceneCenter.x + spread*1.8, sceneCenter.y + spread*0.5, sceneCenter.z + spread*1.8);
      controls.update();

      data.bones.forEach((bone: any) => {
        const geom = new THREE.BufferGeometry();
        
        const positions = new Float32Array(bone.vertices.length * 3);
        for (let i = 0; i < bone.vertices.length; i++) {
          positions[i*3]   = bone.vertices[i][0];
          positions[i*3+1] = bone.vertices[i][1];
          positions[i*3+2] = bone.vertices[i][2];
        }
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        let mesh: THREE.Object3D;
        if (bone.indices && bone.indices.length > 0) {
            geom.setIndex(bone.indices);
            geom.computeVertexNormals();
            const mat = new THREE.MeshStandardMaterial({
              color: bone.color,
              roughness: 0.5,
              metalness: 0.1,
              transparent: true,
              opacity: 0.4,
              side: THREE.DoubleSide
            });
            mesh = new THREE.Mesh(geom, mat);
        } else {
            const mat = new THREE.PointsMaterial({
              color: bone.color,
              size: 2.0,
              sizeAttenuation: true
            });
            mesh = new THREE.Points(geom, mat);
        }
        
        scene.add(mesh);

        // Add Local Origin Sphere (Gold)
        const originGeom = new THREE.SphereGeometry(3, 16, 16);
        const originMat = new THREE.MeshBasicMaterial({ color: 0xffd700, depthTest: false }); // Gold
        const originSphere = new THREE.Mesh(originGeom, originMat);
        originSphere.renderOrder = 1001;
        mesh.add(originSphere);

        // Add Origin Label
        const canvas = document.createElement('canvas');
        canvas.width = 256; canvas.height = 64;
        const spriteMap = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({ map: spriteMap, depthTest: false });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.scale.set(60, 15, 1);
        sprite.renderOrder = 1002;
        scene.add(sprite);

        // Store Bone References and Origin Markers
        if (bone.label === "Thorax") {
          thoraxMesh.value = mesh as THREE.Mesh;
          originMarkers.thorax = originSphere;
          originLabels.thorax = sprite;
        } else if (bone.label === "R Clavicle") {
          clavicleMeshes.right = mesh as THREE.Mesh;
          originMarkers.clavicle.right = originSphere;
          originLabels.clavicle.right = sprite;
          initialQuats.clavicle.right.copy(mesh.quaternion);
          initialPositions.clavicle.right.copy(mesh.position);
        } else if (bone.label === "L Clavicle") {
          clavicleMeshes.left = mesh as THREE.Mesh;
          originMarkers.clavicle.left = originSphere;
          originLabels.clavicle.left = sprite;
          initialQuats.clavicle.left.copy(mesh.quaternion);
          initialPositions.clavicle.left.copy(mesh.position);
        } else if (bone.label === "R Scapula") {
          scapulaMeshes.right = mesh as THREE.Mesh;
          originMarkers.scapula.right = originSphere;
          originLabels.scapula.right = sprite;
          initialQuats.scapula.right.copy(mesh.quaternion);
          initialPositions.scapula.right.copy(mesh.position);
        } else if (bone.label === "L Scapula") {
          scapulaMeshes.left = mesh as THREE.Mesh;
          originMarkers.scapula.left = originSphere;
          originLabels.scapula.left = sprite;
          initialQuats.scapula.left.copy(mesh.quaternion);
          initialPositions.scapula.left.copy(mesh.position);
        } else if (bone.label === "R Humerus") {
          humerusMeshes.right = mesh as THREE.Mesh;
          originMarkers.humerus.right = originSphere;
          originLabels.humerus.right = sprite;
          initialQuats.humerus.right.copy(mesh.quaternion);
          initialPositions.humerus.right.copy(mesh.position);
        } else if (bone.label === "L Humerus") {
          humerusMeshes.left = mesh as THREE.Mesh;
          originMarkers.humerus.left = originSphere;
          originLabels.humerus.left = sprite;
          initialQuats.humerus.left.copy(mesh.quaternion);
          initialPositions.humerus.left.copy(mesh.position);
        }
      });

      // Extract Joint Data (Full S-A-G Chain)
      if (data.isb_joints) {
        ['right', 'left'].forEach((side) => {
          const jointData = data.isb_joints[side];
          if (jointData) {
            jointPivots[side as 'right'|'left'].sc.set(jointData.sc[0], jointData.sc[1], jointData.sc[2]);
            jointPivots[side as 'right'|'left'].ac.set(jointData.ac[0], jointData.ac[1], jointData.ac[2]);
            jointPivots[side as 'right'|'left'].gh.set(jointData.gh[0], jointData.gh[1], jointData.gh[2]);
            
            const a = jointData.angles;
            if (side === 'right') r_st_coords.value = { abduction: a[0], elevation: a[1], upward: a[2], winging: 0 };
            else l_st_coords.value = { abduction: a[0], elevation: a[1], upward: a[2], winging: 0 };

            // Initialize Visual Joint Markers (Rigid Attachment Mode)
            const colors = { sc: 0xff4444, ac: 0x44ff44, gh: 0x4444ff };
            ['sc', 'ac', 'gh'].forEach((joint) => {
                // Helper to create diagnostic sphere
                const createMarker = (isParent: boolean) => {
                    const geom = new THREE.SphereGeometry(isParent ? 5.5 : 2.5, 16, 16);
                    const mat = new THREE.MeshBasicMaterial({ 
                        color: colors[joint as 'sc'|'ac'|'gh'], 
                        depthTest: false, 
                        transparent: true, 
                        opacity: isParent ? 0.3 : 1.0,
                        wireframe: isParent 
                    });
                    const m = new THREE.Mesh(geom, mat);
                    m.renderOrder = isParent ? 998 : 999;
                    return m;
                };

                const pMarker = createMarker(true);
                const cMarker = createMarker(false);
                const pivot = jointPivots[side as 'right'|'left'][joint as 'sc'|'ac'|'gh'];

                // Explicit Parent/Child Assignment
                if (joint === 'sc') {
                    if (thoraxMesh.value) {
                         const m = pMarker.clone(); m.position.copy(pivot);
                         thoraxMesh.value.add(m);
                         jointMarkersP[side as 'right'|'left'].sc = m;
                    }
                    if (clavicleMeshes[side as 'right'|'left']) {
                         const m = cMarker.clone(); m.position.copy(pivot);
                         clavicleMeshes[side as 'right'|'left']!.add(m);
                         jointMarkersC[side as 'right'|'left'].sc = m;
                    }
                } else if (joint === 'ac') {
                    if (clavicleMeshes[side as 'right'|'left']) {
                         const m = pMarker.clone(); m.position.copy(pivot);
                         clavicleMeshes[side as 'right'|'left']!.add(m);
                         jointMarkersP[side as 'right'|'left'].ac = m;
                    }
                    if (scapulaMeshes[side as 'right'|'left']) {
                         const m = cMarker.clone(); m.position.copy(pivot);
                         scapulaMeshes[side as 'right'|'left']!.add(m);
                         jointMarkersC[side as 'right'|'left'].ac = m;
                    }
                } else if (joint === 'gh') {
                    if (scapulaMeshes[side as 'right'|'left']) {
                         const m = pMarker.clone(); m.position.copy(pivot);
                         scapulaMeshes[side as 'right'|'left']!.add(m);
                         jointMarkersP[side as 'right'|'left'].gh = m;
                    }
                    if (humerusMeshes[side as 'right'|'left']) {
                         const m = cMarker.clone(); m.position.copy(pivot);
                         humerusMeshes[side as 'right'|'left']!.add(m);
                         jointMarkersC[side as 'right'|'left'].gh = m;
                    }
                }

                // Create Label Sprite (Standalone follows World Pos)
                const canvas = document.createElement('canvas');
                canvas.width = 256; canvas.height = 64;
                const spriteMap = new THREE.CanvasTexture(canvas);
                const spriteMat = new THREE.SpriteMaterial({ map: spriteMap, depthTest: false });
                const sprite = new THREE.Sprite(spriteMat);
                sprite.scale.set(60, 15, 1);
                sprite.renderOrder = 1000;
                scene.add(sprite);
                jointLabels[side as 'right'|'left'][joint as 'sc'|'ac'|'gh'] = sprite;
            });
          }
        });
      }
      // Render Markers
      if (data.markers) {
        data.markers.forEach((marker: any) => {
          const sphereGeom = new THREE.SphereGeometry(6, 32, 32);
          const sphereMat = new THREE.MeshStandardMaterial({ color: marker.color, roughness: 0.2 });
          const sphere = new THREE.Mesh(sphereGeom, sphereMat);
          sphere.position.set(marker.pos[0], marker.pos[1], marker.pos[2]);
          scene.add(sphere);
        });
      }

      // Render Axes (Arrows)
      if (data.axes) {
        data.axes.forEach((axis: any) => {
          const dir = new THREE.Vector3(axis.dir[0], axis.dir[1], axis.dir[2]);
          dir.normalize();
          const origin = new THREE.Vector3(axis.start[0], axis.start[1], axis.start[2]);
          const length = 40;
          const arrowHelper = new THREE.ArrowHelper(dir, origin, length, axis.color, 8, 4);
          scene.add(arrowHelper);
        });
      }

    } catch (err) {
      console.error("Failed to load bones.json", err);
    }

    const updateLabels = (side: 'right' | 'left', joint: 'sc' | 'ac' | 'gh') => {
        const markerC = jointMarkersC[side][joint];
        const label = jointLabels[side][joint];
        if (markerC && label) {
            // Get World Position of the child marker
            const worldPos = new THREE.Vector3();
            markerC.getWorldPosition(worldPos);
            
            label.position.copy(worldPos).add(new THREE.Vector3(15, 12, 0));
            const ctx = (label.material.map as THREE.CanvasTexture).image.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, 256, 64);
                ctx.fillStyle = 'rgba(0,0,0,0.6)';
                ctx.fillRect(0, 0, 256, 64);
                ctx.font = 'bold 22px Inter, sans-serif';
                ctx.fillStyle = joint === 'sc' ? '#ff6666' : (joint === 'ac' ? '#66ff66' : '#80bfff');
                const text = `${joint.toUpperCase()} [${Math.round(worldPos.x)},${Math.round(worldPos.y)},${Math.round(worldPos.z)}]`;
                ctx.fillText(text, 10, 40);
                label.material.map!.needsUpdate = true;
            }
        }
    };

    const updateOriginLabels = () => {
        const up = (label: THREE.Sprite | null, marker: THREE.Mesh | null, name: string) => {
            if (!label || !marker) return;
            const worldPos = new THREE.Vector3();
            marker.getWorldPosition(worldPos);
            label.position.copy(worldPos).add(new THREE.Vector3(-15, -12, 0));
            const ctx = (label.material.map as THREE.CanvasTexture).image.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, 256, 64);
                ctx.fillStyle = 'rgba(0,0,0,0.6)';
                ctx.fillRect(0, 0, 256, 64);
                ctx.font = 'bold 18px Inter, sans-serif';
                ctx.fillStyle = '#FFD700';
                const text = `${name} [${Math.round(worldPos.x)},${Math.round(worldPos.y)},${Math.round(worldPos.z)}]`;
                ctx.fillText(text, 10, 40);
                label.material.map!.needsUpdate = true;
            }
        };

        up(originLabels.thorax, originMarkers.thorax, "Thorax (IJ)");
        ['right', 'left'].forEach(s => {
            const side = s as 'right'|'left';
            up(originLabels.clavicle[side], originMarkers.clavicle[side], `${side.toUpperCase()} Clav Origin`);
            up(originLabels.scapula[side], originMarkers.scapula[side], `${side.toUpperCase()} Scap Origin`);
            up(originLabels.humerus[side], originMarkers.humerus[side], `${side.toUpperCase()} Hum Origin`);
        });
    };

    const updateKinematicChain = (side: 'right' | 'left') => {
      const cMesh = clavicleMeshes[side];
      const sMesh = scapulaMeshes[side];
      const hMesh = humerusMeshes[side];
      if (!cMesh || !sMesh || !hMesh) return;

      const coords = side === 'right' ? r_st_coords.value : l_st_coords.value;
      const pivots = jointPivots[side];
      
      // 1. Reset all bones to Zero-Pose (The ISB-aligned mean mesh)
      const allBones = [cMesh, sMesh, hMesh];
      allBones.forEach(b => {
        b.quaternion.copy(new THREE.Quaternion()); // Identity
        b.position.set(0,0,0); // Relative to world 0
      });

      // 2. Define Joint Rotation Quaternions (Currently focusing on SC Refinement)
      const qSC = new THREE.Quaternion().setFromEuler(new THREE.Euler(
        THREE.MathUtils.degToRad(coords.elevation),
        THREE.MathUtils.degToRad(-coords.abduction), 
        THREE.MathUtils.degToRad(coords.upward),
        'YXZ'
      ));
      
      // Note: We could add AC/GH sliders here. For now, we use Python's identity for internal joints.
      const qAC = new THREE.Quaternion(); 
      const qGH = new THREE.Quaternion();

      // 3. Recursive Transform Application
      
      updateLabels(side, 'sc');

      // B. AC JOINT (Clavicle vs Scapula)
      const acP_World = pivots.ac.clone().sub(pivots.sc).applyQuaternion(qSC).add(pivots.sc); 
      
      sMesh.position.sub(pivots.sc);
      sMesh.position.applyQuaternion(qSC);
      sMesh.position.add(pivots.sc);
      sMesh.quaternion.premultiply(qSC);
      
      // Apply AC rotation to Scapula
      sMesh.position.sub(acP_World);
      sMesh.position.applyQuaternion(qAC);
      sMesh.position.add(acP_World);
      sMesh.quaternion.premultiply(qAC);
      
      updateLabels(side, 'ac');

      // C. GH JOINT (Scapula vs Humerus)
      const ghP_World = pivots.gh.clone().sub(pivots.ac).applyQuaternion(qAC).add(pivots.ac);
      ghP_World.sub(pivots.sc).applyQuaternion(qSC).add(pivots.sc);

      hMesh.position.sub(pivots.sc);
      hMesh.position.applyQuaternion(qSC);
      hMesh.position.add(pivots.sc);
      hMesh.quaternion.premultiply(qSC);

      hMesh.position.sub(acP_World);
      hMesh.position.applyQuaternion(qAC);
      hMesh.position.add(acP_World);
      hMesh.quaternion.premultiply(qAC);

      hMesh.position.sub(ghP_World);
      hMesh.position.applyQuaternion(qGH);
      hMesh.position.add(ghP_World);
      hMesh.quaternion.premultiply(qGH);

      updateLabels(side, 'gh');
      updateOriginLabels();
    };

    const animate = () => {
      requestAnimationFrame(animate);
      updateKinematicChain('right');
      updateKinematicChain('left');
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    window.addEventListener('resize', () => {
      if (viewerContainer.value) {
        const fWidth = viewerContainer.value.clientWidth;
        const fHeight = viewerContainer.value.clientHeight;
        camera.aspect = fWidth / fHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(fWidth, fHeight);
      }
    });
  }
});

async function runPrediction() {
  isPredicting.value = true;
  statusMessage.value = "Starting Python pipeline...";
  statusColor.value = "#ffffff";

  try {
    const result = await invoke("run_prediction", {
      args: {
        sex: sex.value,
        age: age.value,
        height: height.value,
        weight: weight.value,
        r_clav_len: r_clav_len.value,
        r_hum_len: r_hum_len.value,
        r_hum_epi_width: r_hum_epi_width.value,
        anthro_path: anthro_path.value,
        ssm_path: ssm_path.value,
        out_path: out_path.value
      }
    });

    statusMessage.value = "Done: " + result;
    statusColor.value = "#48c774";
  } catch (error) {
    statusMessage.value = "Failed: " + error;
    statusColor.value = "#f14668";
  }

  isPredicting.value = false;
}

async function saveReport() {
  if (isSavingReport.value) return;
  isSavingReport.value = true;
  statusMessage.value = "Generating Clinical Report...";
  statusColor.value = "#ffffff";

  try {
    const result = await invoke("save_refinement_report", {
      args: {
        out_path: out_path.value,
        patient: {
            sex: sex.value,
            age: age.value,
            height: height.value,
            weight: weight.value,
        },
        right_st: r_st_coords.value,
        left_st: l_st_coords.value,
      }
    });
    statusMessage.value = result as string;
    statusColor.value = "#48c774";
  } catch (error) {
    statusMessage.value = "Export Failed: " + error;
    statusColor.value = "#f14668";
  }
  isSavingReport.value = false;
}
</script>

<template>
  <div class="container">
    <div class="left-pane">
       <div class="viewer-wrapper">
          <div class="floating-frame" ref="viewerContainer">
            <!-- Three.js Canvas -->
          </div>
          <div class="frame-reflection"></div>
       </div>
    </div>

    <div class="right-pane">
      <div class="viewer-wrapper">
         <div class="floating-frame right-content">
            <div class="pane-header">
              <h2>{{ isSettingsVisible ? 'Application Settings' : (isKinematicVisible ? 'Kinematic Refinement' : 'Shoulder Predictor') }}</h2>
              <div class="header-actions">
                <button @click="isKinematicVisible = !isKinematicVisible; isSettingsVisible = false" class="icon-btn" :class="{ active: isKinematicVisible }" title="Kinematic Alignment">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 11v8a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h15.5a2.5 2.5 0 0 1 0 5H6"></path><path d="M10 11v8a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-8"></path><path d="M10 11h4"></path></svg>
                </button>
                <button @click="isSettingsVisible = !isSettingsVisible; isKinematicVisible = false" class="icon-btn" :class="{ active: isSettingsVisible }" title="Configurations">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                </button>
              </div>
            </div>

            <div v-if="isSettingsVisible" class="settings-view animate-in">
              <div class="card transparent-card">
                <h3 style="color: #60a5fa">📂 Backend Path Configuration</h3>
                <p class="hint">Configure the internal directories for anatomical processing.</p>
                <label>Anthropometric Data CSV</label>
                <input v-model="anthro_path" class="input-fi" />
                <label>SSM Shape Model Directory</label>
                <input v-model="ssm_path" class="input-fi" />
                <label>Output predicted Mesh (.ply)</label>
                <input v-model="out_path" class="input-fi" />
              </div>
              <button @click="isSettingsVisible = false" class="secondary-btn">Close</button>
            </div>

            <div v-else-if="isKinematicVisible" class="settings-view animate-in">
              <div class="card transparent-card">
                <h3 style="color: #FFA040">🦴 Kinematic Refinement</h3>
                <p class="hint">Adjust Scapulothoracic (ST) joint coordinates around the AC pivot.</p>
                <div class="joint-group">
                  <div class="side-label">Right Shoulder</div>
                  <div class="slider-row">
                    <label><span>Abduction</span> <span>{{ r_st_coords.abduction.toFixed(1) }}°</span></label>
                    <input type="range" v-model.number="r_st_coords.abduction" min="0" max="60" step="0.5" />
                  </div>
                  <div class="slider-row">
                    <label><span>Elevation</span> <span>{{ r_st_coords.elevation.toFixed(1) }}°</span></label>
                    <input type="range" v-model.number="r_st_coords.elevation" min="-15" max="30" step="0.5" />
                  </div>
                  <div class="slider-row">
                    <label><span>Upward Rot</span> <span>{{ r_st_coords.upward.toFixed(1) }}°</span></label>
                    <input type="range" v-model.number="r_st_coords.upward" min="-10" max="45" step="0.5" />
                  </div>
                </div>
                <div class="joint-group" style="margin-top: 20px">
                  <div class="side-label">Left Shoulder</div>
                  <div class="slider-row">
                    <label><span>Abduction</span> <span>{{ l_st_coords.abduction.toFixed(1) }}°</span></label>
                    <input type="range" v-model.number="l_st_coords.abduction" min="0" max="60" step="0.5" />
                  </div>
                  <div class="slider-row">
                    <label><span>Elevation</span> <span>{{ l_st_coords.elevation.toFixed(1) }}°</span></label>
                    <input type="range" v-model.number="l_st_coords.elevation" min="-15" max="30" step="0.5" />
                  </div>
                  <div class="slider-row">
                    <label><span>Upward Rot</span> <span>{{ l_st_coords.upward.toFixed(1) }}°</span></label>
                    <input type="range" v-model.number="l_st_coords.upward" min="-10" max="45" step="0.5" />
                  </div>
                </div>
              </div>
              
              <div class="footer-actions">
                <button :disabled="isSavingReport" @click="saveReport" class="run-btn save-btn">
                  <span v-if="!isSavingReport">📋 Save Clinical Report</span>
                  <span v-else>💾 Exporting...</span>
                </button>
              </div>

              <button @click="isKinematicVisible = false" class="secondary-btn">Close</button>
            </div>

            <div v-else class="main-view animate-in">
              <div class="card transparent-card">
                <h3>🩺 Patient Measurements</h3>
                <div class="grid-compact">
                  <div>
                    <label>Sex (0=Male, 1=Female)</label>
                    <input v-model="sex" class="input-fi" />
                  </div>
                  <div>
                    <label>Age (years)</label>
                    <input v-model="age" class="input-fi" />
                  </div>
                  <div>
                    <label>Height (cm)</label>
                    <input v-model="height" class="input-fi" />
                  </div>
                  <div>
                    <label>Weight (kg)</label>
                    <input v-model="weight" class="input-fi" />
                  </div>
                  <div>
                    <label>R Clavicle Length</label>
                    <input v-model="r_clav_len" class="input-fi" />
                  </div>
                  <div>
                    <label>R Humerus Length</label>
                    <input v-model="r_hum_len" class="input-fi" />
                  </div>
                  <div>
                    <label>R Hum Epicondyle Width</label>
                    <input v-model="r_hum_epi_width" class="input-fi" />
                  </div>
                </div>
              </div>

              <button :disabled="isPredicting" @click="runPrediction" class="run-btn">
                <span v-if="!isPredicting">🚀 Run Prediction Pipeline</span>
                <span v-else>🔄 Executing Model Generation...</span>
              </button>

              <div v-if="statusMessage" class="status-box" :style="{ color: statusColor, borderColor: statusColor }">
                <div class="status-label">Pipeline Output:</div>
                {{ statusMessage }}
              </div>
            </div>
         </div>
      </div>
    </div>
  </div>
</template>

<style>
/* Global resets and seamless background */
html, body, #app {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: radial-gradient(circle at 50% 50%, #1e1e36 0%, #0a0a12 100%);
}
</style>

<style scoped>
.container {
  display: flex;
  height: 100vh;
  width: 100vw;
  background: transparent;
  color: #e0e0e0;
  font-family: 'Inter', 'Outfit', sans-serif;
}
.left-pane {
  flex: 1.5; /* Give visual priority to the model */
  background: transparent;
  overflow: hidden;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}
.viewer-wrapper {
  position: relative;
  width: 90%;
  height: 85%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.floating-frame {
  width: 100%;
  height: 100%;
  background: rgba(15, 15, 26, 0.4);
  border-radius: 24px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 40px 100px -20px rgba(0, 0, 0, 0.8), 
              inset 0 0 20px rgba(0, 0, 0, 0.4);
  overflow: hidden;
  backdrop-filter: blur(8px);
  z-index: 2;
}
.frame-reflection {
  position: absolute;
  top: -10px;
  left: -10px;
  right: -10px;
  bottom: -10px;
  background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, transparent 40%);
  border-radius: 30px;
  z-index: 1;
  pointer-events: none;
}
.right-pane {
  flex: 1;
  padding: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  background: transparent;
  align-items: center;
  justify-content: center;
}
.right-content {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.pane-header {
  padding: 15px 25px;
  background: rgba(26, 26, 46, 0.4);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.main-view, .settings-view {
  padding: 15px 25px;
  overflow: hidden; /* Lock scrolling */
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
}
.animate-in {
  animation: fadeIn 0.3s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
h2 {
  margin: 0;
  color: #ffffff;
  font-size: 1.3rem;
  font-weight: 600;
}
.card {
  background: #1e1e36;
  padding: 18px;
  border-radius: 0; /* Square Corners */
  border: 1px solid #2a2a4a;
}
.transparent-card {
    background: transparent;
    border: none;
    box-shadow: none;
    padding: 0;
}
h3 {
  margin-top: 0;
  color: #48c774;
  font-size: 1rem;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.hint {
  font-size: 0.75rem;
  color: #808090;
  margin-top: -10px;
  margin-bottom: 15px;
}
label {
  display: block;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 8px;
  margin-bottom: 4px;
  color: #a0a0b0;
}
.input-fi {
  width: 100%;
  padding: 7px 10px;
  border-radius: 0; /* Square Corners */
  border: 1px solid #333;
  background: #121220;
  color: #fff;
  transition: border-color 0.2s;
  font-size: 0.9rem;
}
.input-fi:focus {
  outline: none;
  border-color: #48c774;
}
.grid-compact {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}
.icon-btn {
  background: #2a2a4a;
  border: none;
  color: #a0a0b0;
  padding: 10px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}
.icon-btn:hover {
  background: #3e3e66;
  color: #fff;
}
.run-btn {
  background: linear-gradient(135deg, #48c774 0%, #3eaf65 100%);
  color: white;
  border: none;
  padding: 12px;
  border-radius: 0; /* Square Corners */
  font-size: 1rem;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 10px 20px rgba(72, 199, 116, 0.2);
}
.run-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 12px 24px rgba(72, 199, 116, 0.3);
}
.run-btn:disabled {
  background: #2a3a2e;
  cursor: not-allowed;
  opacity: 0.6;
  box-shadow: none;
}
.secondary-btn {
  background: #2a2a4a;
  color: #fff;
  border: 1px solid #333;
  padding: 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}
.secondary-btn:hover {
  background: #333;
}
.status-box {
  margin-top: 5px;
  padding: 12px;
  background-color: #161625;
  border-radius: 0; /* Square Corners */
  border: 1px solid currentColor;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: 'Consolas', monospace;
  font-size: 0.8rem;
  max-height: 120px;
  overflow-y: auto;
}
.status-label {
  font-weight: bold;
  font-size: 0.75rem;
  text-transform: uppercase;
  margin-bottom: 8px;
  opacity: 0.7;
}

/* Kinematic Sliders */
.joint-group {
  padding: 10px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.05);
}
.side-label {
  font-size: 0.8rem;
  font-weight: bold;
  color: #fff;
  margin-bottom: 10px;
  text-transform: uppercase;
}
.slider-row {
  margin-bottom: 8px;
}
.slider-row label {
  margin-top: 0;
  display: flex;
  justify-content: space-between;
}
input[type="range"] {
  width: 100%;
  accent-color: #FFA040;
  background: transparent;
  cursor: pointer;
}

.footer-actions {
  margin-top: 10px;
  display: flex;
  gap: 10px;
}
.save-btn {
  flex: 1;
  background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
  box-shadow: 0 10px 20px rgba(59, 130, 246, 0.2);
}
.save-btn:hover:not(:disabled) {
  box-shadow: 0 12px 24px rgba(59, 130, 246, 0.3);
}
</style>
