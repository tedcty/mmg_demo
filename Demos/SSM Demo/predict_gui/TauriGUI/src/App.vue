<script setup lang="ts">
import { ref, onMounted } from "vue";
import * as THREE from 'three';

// Unique ID for this browser tab — isolates predictions and progress from other sessions
const sessionId = crypto.randomUUID();
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Paths
const anthro_path = ref("/home/trix/Dev/Repo/mmg_demo/mmg_demo/Demos/SSM Demo/predict_gui/Resources/anthro_data.csv");
const ssm_path = ref("/home/trix/Dev/Repo/mmg_demo/mmg_demo/Demos/SSM Demo/predict_gui/Resources/SSM_shape_model_103");
const out_path = ref("/home/trix/Dev/Repo/mmg_demo/mmg_demo/Demos/SSM Demo/predict_gui/Resources/predicted_model.ply");

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
// Prediction progress (0–100), driven by the pipeline's streamed stage
// messages. Shown to all users as the demo-facing feedback while the raw
// "Pipeline Output" text stays dev-only.
const predictionProgress = ref(0);

// Ordered pipeline stages → target progress %. Matched by substring against
// the STATUS messages streamed from the Python pipeline so the bar advances
// as each stage begins. Progress is kept monotonic (only ever increases), so
// stages that get skipped (e.g. cached PLSR training) don't stall the bar.
const PIPELINE_STAGES: { match: string; pct: number }[] = [
  { match: "Initialising",       pct: 5 },
  { match: "Loading PCA model",  pct: 15 },
  { match: "PLSR training",      pct: 30 },
  { match: "Loading PCA shape",  pct: 45 },
  { match: "Running PLSR",       pct: 60 },
  { match: "Reconstructing",     pct: 72 },
  { match: "Using mean mesh",    pct: 80 },
  { match: "Saving output",      pct: 88 },
  { match: "Joint Assembly",     pct: 94 },
];
const isSavingReport = ref(false);
const isSettingsVisible = ref(false);
const isKinematicVisible = ref(false);
// Dev mode — gates developer-only tools (e.g. Run FABRIK). Toggle with
// Ctrl+Shift+D; persisted so it survives reloads. Hidden from demo users.
const isDevMode = ref(localStorage.getItem('ssm_dev_mode') === '1');
// Colour theme — light mode is the default. Persisted so the choice survives
// reloads; only an explicit 'dark' opts out of light.
const isLightMode = ref(localStorage.getItem('ssm_theme') !== 'dark');
const SCENE_BG = { light: 0xe8ecf4, dark: 0x1a1a2e };

function toggleTheme() {
  isLightMode.value = !isLightMode.value;
  localStorage.setItem('ssm_theme', isLightMode.value ? 'light' : 'dark');
  if (globalScene) {
    globalScene.background = new THREE.Color(isLightMode.value ? SCENE_BG.light : SCENE_BG.dark);
  }
}


// Joint Coordinates (ISB Standards)
// Defaults set to the empirically-found neutral upright pose.  Loading the
// prediction with these values produces an anatomical neutral standing pose
// directly; sliders remain available for refinement.
const r_joint_coords = ref({
  sc_abduction: 8.0, sc_elevation: 15.0, sc_upward: 2.0,
  ac_internal: 11.5, ac_upward: 9.5, ac_posterior: 2.5,
  gh_flexion: 11.5, gh_abduction: 20.0, gh_internal: 17.0
});
const l_joint_coords = ref({
  sc_abduction: 8.0, sc_elevation: 15.0, sc_upward: 2.0,
  ac_internal: 11.5, ac_upward: 9.5, ac_posterior: 2.5,
  gh_flexion: 11.5, gh_abduction: 20.0, gh_internal: 17.0
});

// Mesh References for dynamic updates
const thoraxMesh = ref<THREE.Mesh | null>(null);
const clavicleMeshes = { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null };
const scapulaMeshes = { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null };
const subscapMeshes = { right: null as THREE.Points | null, left: null as THREE.Points | null };
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

// Anatomical Landmarks (Point-on-Bone)
const anatomicalMarkers = {
    right: { thorax_sc: null as THREE.Mesh | null | undefined, thorax_ij: null as THREE.Mesh | null | undefined, thorax_px: null as THREE.Mesh | null | undefined, thorax_c7: null as THREE.Mesh | null | undefined, thorax_t8: null as THREE.Mesh | null | undefined, clavicle_sc: null as THREE.Mesh | null | undefined, clavicle_ac: null as THREE.Mesh | null | undefined, scapula_ac: null as THREE.Mesh | null | undefined, scapula_aa: null as THREE.Mesh | null | undefined, scapula_ts: null as THREE.Mesh | null | undefined, scapula_ai: null as THREE.Mesh | null | undefined },
    left:  { thorax_sc: null as THREE.Mesh | null | undefined, thorax_ij: null as THREE.Mesh | null | undefined, thorax_px: null as THREE.Mesh | null | undefined, thorax_c7: null as THREE.Mesh | null | undefined, thorax_t8: null as THREE.Mesh | null | undefined, clavicle_sc: null as THREE.Mesh | null | undefined, clavicle_ac: null as THREE.Mesh | null | undefined, scapula_ac: null as THREE.Mesh | null | undefined, scapula_aa: null as THREE.Mesh | null | undefined, scapula_ts: null as THREE.Mesh | null | undefined, scapula_ai: null as THREE.Mesh | null | undefined }
};
const anatomicalLabels = {
    right: { thorax_sc: null as THREE.Sprite | null | undefined, thorax_ij: null as THREE.Sprite | null | undefined, thorax_px: null as THREE.Sprite | null | undefined, thorax_c7: null as THREE.Sprite | null | undefined, thorax_t8: null as THREE.Sprite | null | undefined, clavicle_sc: null as THREE.Sprite | null | undefined, clavicle_ac: null as THREE.Sprite | null | undefined, scapula_ac: null as THREE.Sprite | null | undefined, scapula_aa: null as THREE.Sprite | null | undefined, scapula_ts: null as THREE.Sprite | null | undefined, scapula_ai: null as THREE.Sprite | null | undefined },
    left:  { thorax_sc: null as THREE.Sprite | null | undefined, thorax_ij: null as THREE.Sprite | null | undefined, thorax_px: null as THREE.Sprite | null | undefined, thorax_c7: null as THREE.Sprite | null | undefined, thorax_t8: null as THREE.Sprite | null | undefined, clavicle_sc: null as THREE.Sprite | null | undefined, clavicle_ac: null as THREE.Sprite | null | undefined, scapula_ac: null as THREE.Sprite | null | undefined, scapula_aa: null as THREE.Sprite | null | undefined, scapula_ts: null as THREE.Sprite | null | undefined, scapula_ai: null as THREE.Sprite | null | undefined }
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
let globalScene: THREE.Scene | null = null;
let globalCamera: THREE.PerspectiveCamera | null = null;
let globalControls: OrbitControls | null = null;
let bonesGroup: THREE.Group | null = null;
let ghostGroup: THREE.Group | null = null; // Group for the mean "ghost" overlap
let isFirstLoad = true;

// Comparison State
const isViewingOriginal = ref(true);
const isOverlapEnabled = ref(false); // New: Overlap Mode state
const hasPrediction = ref(false);
const showGuides = ref(false); // Master toggle: spheres, triangles, muscle/glide areas, labels
const isHighlightsEnabled = ref(true); // Control for glide area visualization
const isNormalsEnabled = ref(false); // Control for surface normals
const isScapularPlaneEnabled = ref(false); // Control for scapular plane
const isLabelsEnabled = ref(true); // Control for coordinate labels
let highlightsGroup: THREE.Group | null = null; 
let scapularPlaneGroup: THREE.Group | null = null; 
let meanModelData: any = null;
let predictedModelData: any = null;

// Reference storage for ghost meshes
const ghostMeshes = {
    thorax: null as THREE.Mesh | null,
    clavicle: { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null },
    scapula: { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null },
    humerus: { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null }
};

async function loadBones(externalData: any = null) {
  if (!globalScene) return;

  try {
    let data;
    if (externalData) {
      data = externalData;
      console.log("Loading bones from injected Rust data...");
    } else {
      // Add cache-busting timestamp to ensure we get the fresh bones.json
      const response = await fetch(`/bones.json?t=${Date.now()}`);
      data = await response.json();
      console.log("Loading bones from public/bones.json...");
    }

    if (!meanModelData) {
      meanModelData = JSON.parse(JSON.stringify(data));
    }
    predictedModelData = data;
    hasPrediction.value = true;
    
    const activeData = isViewingOriginal.value ? meanModelData : predictedModelData;
    const center = activeData.center;
    const spread = activeData.spread || 500;
    
    if (isFirstLoad && globalCamera && globalControls) {
      const sceneCenter = new THREE.Vector3(center[0], center[1], center[2]);
      globalControls.target.copy(sceneCenter);
      globalCamera.position.set(sceneCenter.x + spread*1.8, sceneCenter.y + spread*0.5, sceneCenter.z + spread*1.8);
      globalControls.update();
      isFirstLoad = false;
    }

    // Determine if we need to recreate or just update
    const getMesh = (label: string) => {
        if (label === "Thorax") return thoraxMesh.value;
        if (label === "R Clavicle") return clavicleMeshes.right;
        if (label === "L Clavicle") return clavicleMeshes.left;
        if (label === "R Scapula") return scapulaMeshes.right;
        if (label === "L Scapula") return scapulaMeshes.left;
        if (label === "R Humerus") return humerusMeshes.right;
        if (label === "L Humerus") return humerusMeshes.left;
        return null;
    };

    // Always do full recreation to ensure scapular planes, markers, and
    // landmarks update correctly after FABRIK steps.
    const needsFullRecreation = true;

    if (needsFullRecreation) {
      // --- FULL RECREATION ---
      if (bonesGroup) {
        globalScene.remove(bonesGroup);
        bonesGroup.traverse((obj) => {
          if (obj instanceof THREE.Mesh || obj instanceof THREE.Points) {
            obj.geometry.dispose();
            if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
            else obj.material.dispose();
          }
        });
      }
      bonesGroup = new THREE.Group();
      globalScene.add(bonesGroup);

      activeData.bones.forEach((bone: any) => {
        const geom = new THREE.BufferGeometry();
        const positions = new Float32Array(bone.vertices.length * 3);
        for (let i = 0; i < bone.vertices.length; i++) {
          positions[i*3] = bone.vertices[i][0];
          positions[i*3+1] = bone.vertices[i][1];
          positions[i*3+2] = bone.vertices[i][2];
        }
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        let mesh: THREE.Mesh | THREE.Points;
        if (bone.indices && bone.indices.length > 0) {
            geom.setIndex(bone.indices);
            geom.computeVertexNormals();
            const opac = bone.label === "Thorax" ? 0.1 : 0.55;
            const mat = new THREE.MeshStandardMaterial({
              color: isViewingOriginal.value ? "#88aaff" : bone.color,
              roughness: 0.5, metalness: 0.1, transparent: true, opacity: opac, side: THREE.DoubleSide
            });
            mesh = new THREE.Mesh(geom, mat);
        } else {
            const mat = new THREE.PointsMaterial({ color: bone.color, size: 2.0, sizeAttenuation: true });
            mesh = new THREE.Points(geom, mat);
        }
        
        bonesGroup!.add(mesh);

        // Origin marker sphere + label — part of the visual guides
        let originSphere: THREE.Mesh | null = null;
        let sprite: THREE.Sprite | null = null;
        if (showGuides.value) {
            const originGeom = new THREE.SphereGeometry(3, 16, 16);
            const originMat = new THREE.MeshBasicMaterial({ color: 0xffd700, depthTest: false });
            originSphere = new THREE.Mesh(originGeom, originMat);
            originSphere.renderOrder = 1001;
            if (bone.origin) {
                originSphere.position.set(bone.origin[0], bone.origin[1], bone.origin[2]);
            }
            bonesGroup!.add(originSphere); // Add to bonesGroup directly so it stays at world pos

            const canvas = document.createElement('canvas');
            canvas.width = 1024; canvas.height = 256;
            const spriteMap = new THREE.CanvasTexture(canvas);
            spriteMap.anisotropy = 16;
            const spriteMat = new THREE.SpriteMaterial({ map: spriteMap, depthTest: false });
            sprite = new THREE.Sprite(spriteMat);
            sprite.scale.set(80, 20, 1);
            sprite.renderOrder = 1002;
            bonesGroup!.add(sprite);
        }

        if (bone.label === "Thorax") { thoraxMesh.value = mesh as THREE.Mesh; originMarkers.thorax = originSphere; originLabels.thorax = sprite; }
        else if (bone.label === "R Clavicle") { clavicleMeshes.right = mesh as THREE.Mesh; originMarkers.clavicle.right = originSphere; originLabels.clavicle.right = sprite; initialQuats.clavicle.right.copy(mesh.quaternion); initialPositions.clavicle.right.copy(mesh.position); }
        else if (bone.label === "L Clavicle") { clavicleMeshes.left = mesh as THREE.Mesh; originMarkers.clavicle.left = originSphere; originLabels.clavicle.left = sprite; initialQuats.clavicle.left.copy(mesh.quaternion); initialPositions.clavicle.left.copy(mesh.position); }
        else if (bone.label === "R Scapula") { scapulaMeshes.right = mesh as THREE.Mesh; originMarkers.scapula.right = originSphere; originLabels.scapula.right = sprite; initialQuats.scapula.right.copy(mesh.quaternion); initialPositions.scapula.right.copy(mesh.position); }
        else if (bone.label === "L Scapula") { scapulaMeshes.left = mesh as THREE.Mesh; originMarkers.scapula.left = originSphere; originLabels.scapula.left = sprite; initialQuats.scapula.left.copy(mesh.quaternion); initialPositions.scapula.left.copy(mesh.position); }
        else if (bone.label === "R Subscapularis") { subscapMeshes.right = mesh as THREE.Points; }
        else if (bone.label === "L Subscapularis") { subscapMeshes.left = mesh as THREE.Points; }
        else if (bone.label === "R Humerus") { humerusMeshes.right = mesh as THREE.Mesh; originMarkers.humerus.right = originSphere; originLabels.humerus.right = sprite; initialQuats.humerus.right.copy(mesh.quaternion); initialPositions.humerus.right.copy(mesh.position); }
        else if (bone.label === "L Humerus") { humerusMeshes.left = mesh as THREE.Mesh; originMarkers.humerus.left = originSphere; originLabels.humerus.left = sprite; initialQuats.humerus.left.copy(mesh.quaternion); initialPositions.humerus.left.copy(mesh.position); }
      });

      // Post-process: parent subscapularis to scapula so it moves with it
      if (subscapMeshes.right && scapulaMeshes.right) { scapulaMeshes.right.add(subscapMeshes.right); }
      if (subscapMeshes.left && scapulaMeshes.left) { scapulaMeshes.left.add(subscapMeshes.left); }
    } else {
      // --- SMOOTH UPDATE ---
      activeData.bones.forEach((bone: any) => {
          const mesh = getMesh(bone.label);
          if (mesh) {
              const posAttr = mesh.geometry.attributes.position;
              for (let i = 0; i < bone.vertices.length; i++) {
                  posAttr.setXYZ(i, bone.vertices[i][0], bone.vertices[i][1], bone.vertices[i][2]);
              }
              posAttr.needsUpdate = true;
              mesh.geometry.computeVertexNormals();
              if (mesh.material instanceof THREE.MeshStandardMaterial) {
                  mesh.material.color.set(isViewingOriginal.value ? "#88aaff" : bone.color);
                  mesh.material.opacity = bone.label === "Thorax" ? 0.1 : 0.55;
              }
          }
      });
    }

    // Always update joints, markers, and landmarks regardless of update mode
    if (activeData.isb_joints) {
      ['right', 'left'].forEach((side) => {
        const jointData = activeData.isb_joints[side];
        if (jointData) {
          jointPivots[side as 'right'|'left'].sc.set(jointData.sc[0], jointData.sc[1], jointData.sc[2]);
          jointPivots[side as 'right'|'left'].ac.set(jointData.ac[0], jointData.ac[1], jointData.ac[2]);
          jointPivots[side as 'right'|'left'].gh.set(jointData.gh[0], jointData.gh[1], jointData.gh[2]);
          
          if (needsFullRecreation && showGuides.value) {
            const colors = { sc: 0xff4444, ac: 0x44ff44, gh: 0x4444ff };
            ['sc', 'ac', 'gh'].forEach((joint) => {
                const pMarker = new THREE.Mesh(new THREE.SphereGeometry(5.5, 16, 16), new THREE.MeshBasicMaterial({ color: colors[joint as 'sc'|'ac'|'gh'], depthTest: false, transparent: true, opacity: 0.3, wireframe: true }));
                const cMarker = new THREE.Mesh(new THREE.SphereGeometry(2.5, 16, 16), new THREE.MeshBasicMaterial({ color: colors[joint as 'sc'|'ac'|'gh'], depthTest: false }));
                const pivot = jointPivots[side as 'right'|'left'][joint as 'sc'|'ac'|'gh'];

                if (joint === 'sc') {
                    if (thoraxMesh.value) { const m = pMarker.clone(); m.position.copy(pivot); thoraxMesh.value.add(m); jointMarkersP[side as 'right'|'left'].sc = m; }
                    if (clavicleMeshes[side as 'right'|'left']) { const m = cMarker.clone(); m.position.copy(pivot); clavicleMeshes[side as 'right'|'left']!.add(m); jointMarkersC[side as 'right'|'left'].sc = m; }
                } else if (joint === 'ac') {
                    if (clavicleMeshes[side as 'right'|'left']) { const m = pMarker.clone(); m.position.copy(pivot); clavicleMeshes[side as 'right'|'left']!.add(m); jointMarkersP[side as 'right'|'left'].ac = m; }
                    if (scapulaMeshes[side as 'right'|'left']) { const m = cMarker.clone(); m.position.copy(pivot); scapulaMeshes[side as 'right'|'left']!.add(m); jointMarkersC[side as 'right'|'left'].ac = m; }
                } else if (joint === 'gh') {
                    if (scapulaMeshes[side as 'right'|'left']) { const m = pMarker.clone(); m.position.copy(pivot); scapulaMeshes[side as 'right'|'left']!.add(m); jointMarkersP[side as 'right'|'left'].gh = m; }
                    if (humerusMeshes[side as 'right'|'left']) { const m = cMarker.clone(); m.position.copy(pivot); humerusMeshes[side as 'right'|'left']!.add(m); jointMarkersC[side as 'right'|'left'].gh = m; }
                }
            });
          }
        }
      });
    }

    if (needsFullRecreation && showGuides.value && activeData.markers) {
        activeData.markers.forEach((marker: any) => {
            const sphere = new THREE.Mesh(new THREE.SphereGeometry(6, 32, 32), new THREE.MeshStandardMaterial({ color: marker.color, roughness: 0.2 }));
            sphere.position.set(marker.pos[0], marker.pos[1], marker.pos[2]);
            bonesGroup!.add(sphere);
        });
    }

    // --- SCAPULAR PLANE TRIANGLES ---
    if (needsFullRecreation && showGuides.value && activeData.scapular_planes) {
        ['right', 'left'].forEach((side) => {
            const plane = activeData.scapular_planes[side];
            if (!plane || !plane.aa || !plane.ts || !plane.ai) return;
            
            const aa = new THREE.Vector3().fromArray(plane.aa);
            const ts = new THREE.Vector3().fromArray(plane.ts);
            const ai = new THREE.Vector3().fromArray(plane.ai);
            const cen = new THREE.Vector3().fromArray(plane.centroid);
            
            // Triangle mesh (semi-transparent)
            const triGeom = new THREE.BufferGeometry();
            const vertices = new Float32Array([
                aa.x, aa.y, aa.z,
                ts.x, ts.y, ts.z,
                ai.x, ai.y, ai.z,
            ]);
            triGeom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            triGeom.computeVertexNormals();
            const triMat = new THREE.MeshBasicMaterial({
                color: side === 'right' ? 0xFF6600 : 0xFFCC00,
                transparent: true,
                opacity: 0.35,
                side: THREE.DoubleSide,
                depthTest: false,
            });
            const triMesh = new THREE.Mesh(triGeom, triMat);
            triMesh.renderOrder = 999;
            bonesGroup!.add(triMesh);

            // Triangle edges (wireframe outline)
            const edgeGeom = new THREE.BufferGeometry().setFromPoints([aa, ts, ai, aa]);
            const edgeMat = new THREE.LineBasicMaterial({
                color: side === 'right' ? 0xFF8800 : 0xFFDD44,
                linewidth: 2,
                depthTest: false,
            });
            const edgeLine = new THREE.Line(edgeGeom, edgeMat);
            edgeLine.renderOrder = 1000;
            bonesGroup!.add(edgeLine);

            // Normal arrow from centroid (use pre-computed corrected normal)
            let normal: THREE.Vector3;
            if (plane.normal) {
                normal = new THREE.Vector3().fromArray(plane.normal);
            } else {
                // Fallback: compute from cross product
                const v1 = new THREE.Vector3().subVectors(aa, ts);
                const v2 = new THREE.Vector3().subVectors(ai, ts);
                normal = new THREE.Vector3().crossVectors(v1, v2).normalize();
            }
            const arrow = new THREE.ArrowHelper(normal, cen, 30, side === 'right' ? 0xFF4400 : 0xFFBB00, 8, 4);
            (arrow as any).renderOrder = 1001;
            bonesGroup!.add(arrow);
        });
    }


    if (needsFullRecreation && showGuides.value && activeData.anatomical_landmarks) {
      ['right', 'left'].forEach((side) => {
        const lms = activeData.anatomical_landmarks[side];
        const colors = { thorax: 0x00FFFF, clavicle: 0xFFA500, scapula: 0xFFFF00 };
        const s_t = side as 'right' | 'left';
        
        const createAnthroMarker = (pos: number[], color: number, parent: THREE.Mesh | null) => {
          if (!parent) return { marker: null, label: null };
          const mesh = new THREE.Mesh(new THREE.SphereGeometry(3.5, 16, 16), new THREE.MeshBasicMaterial({ color: color, depthTest: false }));
          mesh.position.set(pos[0], pos[1], pos[2]);
          mesh.renderOrder = 1010;
          parent.add(mesh);
          const canvas = document.createElement('canvas');
          canvas.width = 1024;
          canvas.height = 256;
          const texture = new THREE.CanvasTexture(canvas);
          texture.anisotropy = 16; 
          const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
          sprite.scale.set(80, 20, 1);
          sprite.renderOrder = 1011;
          bonesGroup!.add(sprite);
          return { marker: mesh, sprite: sprite };
        };

        const r1 = createAnthroMarker(lms.thorax_sc, colors.thorax, thoraxMesh.value);
        anatomicalMarkers[s_t].thorax_sc = r1.marker; anatomicalLabels[s_t].thorax_sc = r1.sprite;
        const ri = createAnthroMarker(lms.thorax_ij, colors.thorax, thoraxMesh.value);
        anatomicalMarkers[s_t].thorax_ij = ri.marker; anatomicalLabels[s_t].thorax_ij = ri.sprite;
        const rp = createAnthroMarker(lms.thorax_px, colors.thorax, thoraxMesh.value);
        anatomicalMarkers[s_t].thorax_px = rp.marker; anatomicalLabels[s_t].thorax_px = rp.sprite;
        const rc7 = createAnthroMarker(lms.thorax_c7, colors.thorax, thoraxMesh.value);
        anatomicalMarkers[s_t].thorax_c7 = rc7.marker; anatomicalLabels[s_t].thorax_c7 = rc7.sprite;
        const rt8 = createAnthroMarker(lms.thorax_t8, colors.thorax, thoraxMesh.value);
        anatomicalMarkers[s_t].thorax_t8 = rt8.marker; anatomicalLabels[s_t].thorax_t8 = rt8.sprite;

        const r2 = createAnthroMarker(lms.clavicle_sc, colors.clavicle, clavicleMeshes[s_t]);
        anatomicalMarkers[s_t].clavicle_sc = r2.marker; anatomicalLabels[s_t].clavicle_sc = r2.sprite;
        const r3 = createAnthroMarker(lms.clavicle_ac, colors.clavicle, clavicleMeshes[s_t]);
        anatomicalMarkers[s_t].clavicle_ac = r3.marker; anatomicalLabels[s_t].clavicle_ac = r3.sprite;
        const r4 = createAnthroMarker(lms.scapula_ac, colors.scapula, scapulaMeshes[s_t]);
        anatomicalMarkers[s_t].scapula_ac = r4.marker; anatomicalLabels[s_t].scapula_ac = r4.sprite;
        const raa = createAnthroMarker(lms.scapula_aa, colors.scapula, scapulaMeshes[s_t]);
        anatomicalMarkers[s_t].scapula_aa = raa.marker; anatomicalLabels[s_t].scapula_aa = raa.sprite;
        const rts = createAnthroMarker(lms.scapula_ts, colors.scapula, scapulaMeshes[s_t]);
        anatomicalMarkers[s_t].scapula_ts = rts.marker; anatomicalLabels[s_t].scapula_ts = rts.sprite;
        const rai = createAnthroMarker(lms.scapula_ai, colors.scapula, scapulaMeshes[s_t]);
        anatomicalMarkers[s_t].scapula_ai = rai.marker; anatomicalLabels[s_t].scapula_ai = rai.sprite;
      });
    }

    // --- THORAX POSTERIOR HIGHLIGHTS ---
    if (showGuides.value && isHighlightsEnabled.value) {
        if (highlightsGroup) {
            globalScene.remove(highlightsGroup);
            highlightsGroup.traverse((obj) => { if (obj instanceof THREE.Mesh) { obj.geometry.dispose(); (obj.material as THREE.Material).dispose(); } });
        }
        highlightsGroup = new THREE.Group();
        globalScene.add(highlightsGroup);

        // Create highlight patches that are perfectly flush with the thorax mesh
        // We do this by extracting a subset of the thorax vertices/faces in the glide zone
        const createFlushHighlight = (side: 'right' | 'left') => {
            const thoraxB = activeData.bones.find((b: any) => b.label === "Thorax");
            if (!thoraxB || !thoraxMesh.value) return;

            const lms = activeData.anatomical_landmarks[side];
            if (!lms || !lms.thorax_c7 || !lms.thorax_t8) return;

            const c7 = new THREE.Vector3().fromArray(lms.thorax_c7);
            const t8 = new THREE.Vector3().fromArray(lms.thorax_t8);
            
            // Define the zone center: Lateral to the spinal line between C7 and T8
            const spineMid = new THREE.Vector3().lerpVectors(c7, t8, 0.45);
            const zoneCenter = spineMid.clone();
            // Move lateral to the spine (World Z is Left +, so Right is -)
            zoneCenter.z += (side === 'right' ? -95 : 95);
            
            const verts = thoraxB.vertices;
            const indices = thoraxB.indices;
            
            const subIndices: number[] = [];
            const radiusSq = 110 * 110; 

            // Iterate through thorax faces and keep those near the zone center
            for (let i = 0; i < indices.length; i += 3) {
                const v1 = verts[indices[i]];
                const dx = v1[0] - zoneCenter.x;
                const dy = v1[1] - zoneCenter.y;
                const dz = v1[2] - zoneCenter.z;
                
                if (dx*dx + dy*dy + dz*dz < radiusSq) {
                    subIndices.push(indices[i], indices[i+1], indices[i+2]);
                }
            }

            if (subIndices.length === 0) return;

            const geom = new THREE.BufferGeometry();
            // Flattening huge vertex arrays can be expensive, but needed for BufferAttribute
            const flatVerts = new Float32Array(verts.length * 3);
            for(let j=0; j<verts.length; j++) {
                flatVerts[j*3] = verts[j][0];
                flatVerts[j*3+1] = verts[j][1];
                flatVerts[j*3+2] = verts[j][2];
            }

            geom.setAttribute('position', new THREE.BufferAttribute(flatVerts, 3));
            geom.setIndex(subIndices);
            geom.computeVertexNormals();

            const mat = new THREE.MeshBasicMaterial({
                color: side === 'right' ? 0x00ff00 : 0x0000ff, // Right: Green, Left: Blue
                transparent: true,
                opacity: 0.6,
                side: THREE.DoubleSide,
                polygonOffset: true,
                polygonOffsetFactor: -1,
                polygonOffsetUnits: -1
            });

            const mesh = new THREE.Mesh(geom, mat);
            mesh.name = `highlight_${side}`;
            
            // Sync with parent thorax
            mesh.position.copy(thoraxMesh.value.position);
            mesh.quaternion.copy(thoraxMesh.value.quaternion);
            mesh.scale.copy(thoraxMesh.value.scale);

            highlightsGroup!.add(mesh);

            // --- ADD NORMALS VISUALIZATION ---
            if (isNormalsEnabled.value) {
                // Sample a few normals to show surface direction
                const posAttr = geom.getAttribute('position');
                const normAttr = geom.getAttribute('normal');
                const step = 20; // Every 20th vertex to avoid clutter
                for (let i = 0; i < subIndices.length; i += step * 3) {
                    const vIdx = subIndices[i];
                    const pos = new THREE.Vector3().fromBufferAttribute(posAttr, vIdx);
                    const norm = new THREE.Vector3().fromBufferAttribute(normAttr, vIdx);
                    
                    const arrow = new THREE.ArrowHelper(
                        norm, 
                        pos, 
                        15, // Length 
                        side === 'right' ? 0x00ff00 : 0x0000ff, // Color matches patch
                        4,  // Head length
                        2   // Head width
                    );
                    // Sync arrow with thorax transform if needed (since it's added to highlightsGroup)
                    highlightsGroup!.add(arrow);
                }
            }
        };

        createFlushHighlight('right');
        createFlushHighlight('left');
    } else if (highlightsGroup) {
        globalScene.remove(highlightsGroup);
        highlightsGroup = null;
    }

    // --- SCAPULAR PLANE VISUALIZATION ---
    if (showGuides.value && isScapularPlaneEnabled.value && activeData.anatomical_landmarks) {
        if (scapularPlaneGroup) {
            globalScene.remove(scapularPlaneGroup);
            scapularPlaneGroup.traverse((obj) => { if (obj instanceof THREE.Mesh) { obj.geometry.dispose(); (obj.material as THREE.Material).dispose(); } });
        }
        scapularPlaneGroup = new THREE.Group();
        globalScene.add(scapularPlaneGroup);

        ['right', 'left'].forEach((side) => {
            const lms = activeData.anatomical_landmarks[side];
            if (!lms.scapula_aa || !lms.scapula_ts || !lms.scapula_ai) return;

            const aa = new THREE.Vector3().fromArray(lms.scapula_aa);
            const ts = new THREE.Vector3().fromArray(lms.scapula_ts);
            const ai = new THREE.Vector3().fromArray(lms.scapula_ai);

            // Centroid
            const centroid = new THREE.Vector3().add(aa).add(ts).add(ai).multiplyScalar(1/3);

            // Plane Geometry
            const geom = new THREE.BufferGeometry();
            const vertices = new Float32Array([
                aa.x, aa.y, aa.z,
                ts.x, ts.y, ts.z,
                ai.x, ai.y, ai.z
            ]);
            geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            geom.computeVertexNormals();

            const mat = new THREE.MeshBasicMaterial({ color: 0xffff00, side: THREE.DoubleSide, transparent: true, opacity: 0.3 });
            const plane = new THREE.Mesh(geom, mat);
            scapularPlaneGroup!.add(plane);

            // Normal Arrow
            const v1 = new THREE.Vector3().subVectors(aa, ts);
            const v2 = new THREE.Vector3().subVectors(ai, ts);
            const normal = new THREE.Vector3().crossVectors(v1, v2).normalize();
            
            // Ensure normal points posteriorly. In the thorax JCS the
            // anterior–posterior axis is X (posterior = −X). Using Z here
            // worked for the right scapula by coincidence but inverted the
            // left, because the AA-TS-AI triangle has opposite chirality
            // between sides.
            if (normal.x > 0) normal.multiplyScalar(-1);

            const arrow = new THREE.ArrowHelper(normal, centroid, 40, 0xffff00, 8, 4);
            scapularPlaneGroup!.add(arrow);
        });
    } else if (scapularPlaneGroup) {
        globalScene.remove(scapularPlaneGroup);
        scapularPlaneGroup = null;
    }

    // --- GHOST OVERLAP LOGIC ---
    if (isOverlapEnabled.value && meanModelData) {
        if (ghostGroup) {
            globalScene.remove(ghostGroup);
            ghostGroup.traverse((obj) => { if (obj instanceof THREE.Mesh) { obj.geometry.dispose(); (obj.material as THREE.Material).dispose(); } });
        }
        ghostGroup = new THREE.Group();
        globalScene.add(ghostGroup);

        meanModelData.bones.forEach((bone: any) => {
            const geom = new THREE.BufferGeometry();
            const positions = new Float32Array(bone.vertices.length * 3);
            for (let i = 0; i < bone.vertices.length; i++) {
                positions[i*3] = bone.vertices[i][0];
                positions[i*3+1] = bone.vertices[i][1];
                positions[i*3+2] = bone.vertices[i][2];
            }
            geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            if (bone.indices && bone.indices.length > 0) {
                geom.setIndex(bone.indices);
                geom.computeVertexNormals();
                const mat = new THREE.MeshStandardMaterial({ color: "#4facfe", transparent: true, opacity: 0.15, wireframe: true, depthWrite: false });
                const mesh = new THREE.Mesh(geom, mat);
                ghostGroup!.add(mesh);
                if (bone.label === "Thorax") ghostMeshes.thorax = mesh;
                else if (bone.label === "R Clavicle") ghostMeshes.clavicle.right = mesh;
                else if (bone.label === "L Clavicle") ghostMeshes.clavicle.left = mesh;
                else if (bone.label === "R Scapula") ghostMeshes.scapula.right = mesh;
                else if (bone.label === "L Scapula") ghostMeshes.scapula.left = mesh;
                else if (bone.label === "R Humerus") ghostMeshes.humerus.right = mesh;
                else if (bone.label === "L Humerus") ghostMeshes.humerus.left = mesh;
            }
        });
    } else if (ghostGroup) {
        globalScene.remove(ghostGroup);
        ghostGroup = null;
    }

  } catch (err) {
    console.error("Failed to load bones.json", err);
  }
}

onMounted(async () => {
  // Stream progress messages from the Python pipeline via SSE
  const evtSource = new EventSource(`/api/progress?session=${sessionId}`);
  evtSource.onmessage = (event) => {
    const text = event.data as string;
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

    // Advance the progress bar to the furthest stage seen so far.
    if (isPredicting.value) {
      const stage = PIPELINE_STAGES.find((s) => text.includes(s.match));
      if (stage) predictionProgress.value = Math.max(predictionProgress.value, stage.pct);
    }
  };

  // Initialize Three.js natively
  if (viewerContainer.value) {
    const width = viewerContainer.value.clientWidth;
    const height = viewerContainer.value.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(isLightMode.value ? SCENE_BG.light : SCENE_BG.dark);

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

    globalScene = scene;
    globalCamera = camera;
    globalControls = controls;
    loadBones();

    const updateLabels = (side: 'right' | 'left', joint: 'sc' | 'ac' | 'gh') => {
        const markerC = jointMarkersC[side][joint];
        const label = jointLabels[side][joint];
        if (markerC && label) {
            label.visible = isLabelsEnabled.value;
            if (!label.visible) return;
            // Get World Position of the child marker
            const worldPos = new THREE.Vector3();
            markerC.getWorldPosition(worldPos);
            
            label.position.copy(worldPos).add(new THREE.Vector3(15, 12, 0));
            const canvas = (label.material.map as THREE.CanvasTexture).image;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, 1024, 256);
                ctx.fillStyle = 'rgba(0,0,0,0.6)';
                ctx.fillRect(0, 0, 1024, 256);
                ctx.font = 'bold 80px Inter, sans-serif';
                ctx.fillStyle = joint === 'sc' ? '#ff6666' : (joint === 'ac' ? '#66ff66' : '#80bfff');
                const text = `${joint.toUpperCase()} [${Math.round(worldPos.x)},${Math.round(worldPos.y)},${Math.round(worldPos.z)}]`;
                ctx.fillText(text, 40, 160);
                label.material.map!.needsUpdate = true;
            }
        }
    };

    const updateOriginLabels = () => {
        const up = (label: THREE.Sprite | null, marker: THREE.Mesh | null, name: string) => {
            if (!label || !marker) return;
            label.visible = isLabelsEnabled.value;
            if (!label.visible) return;
            const worldPos = new THREE.Vector3();
            marker.getWorldPosition(worldPos);
            label.position.copy(worldPos).add(new THREE.Vector3(-15, -12, 0));
            const canvas = (label.material.map as THREE.CanvasTexture).image;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, 1024, 256);
                ctx.fillStyle = 'rgba(0,0,0,0.6)';
                ctx.fillRect(0, 0, 1024, 256);
                ctx.font = 'bold 64px Inter, sans-serif';
                ctx.fillStyle = '#FFD700';
                const text = `${name} [${Math.round(worldPos.x)},${Math.round(worldPos.y)},${Math.round(worldPos.z)}]`;
                ctx.fillText(text, 40, 160);
                label.material.map!.needsUpdate = true;
            }
        };

        up(originLabels.thorax, originMarkers.thorax, "Thorax (IJ)");
        ['right', 'left'].forEach(s => {
            const side = s as 'right'|'left';
            up(originLabels.clavicle[side], originMarkers.clavicle[side], `${side.toUpperCase()} Clav Origin`);
            up(originLabels.scapula[side], originMarkers.scapula[side], `${side.toUpperCase()} Scap Origin`);
            up(originLabels.humerus[side], originMarkers.humerus[side], `${side.toUpperCase()} Hum Origin`);

            // Update Anatomical Labels
            const lms = anatomicalLabels[side];
            const markers = anatomicalMarkers[side];
            const boneNames = { 
                thorax_sc: "Thorax", thorax_ij: "Thorax", thorax_px: "Thorax", thorax_c7: "Thorax", thorax_t8: "Thorax",
                clavicle_sc: "Clavicle", clavicle_ac: "Clavicle", scapula_ac: "Scapula",
                scapula_aa: "Scapula", scapula_ts: "Scapula", scapula_ai: "Scapula"
            };
            const jointNames = { 
                thorax_sc: "SC", thorax_ij: "IJ", thorax_px: "PX", thorax_c7: "C7", thorax_t8: "T8",
                clavicle_sc: "SC", clavicle_ac: "AC", scapula_ac: "AC",
                scapula_aa: "AA", scapula_ts: "TS", scapula_ai: "AI"
            };
            const colors = { 
                thorax_sc: "#00FFFF", thorax_ij: "#FFFF00", thorax_px: "#FFFF00", thorax_c7: "#FFFF00", thorax_t8: "#FFFF00",
                clavicle_sc: "#FFA500", clavicle_ac: "#FFA500", scapula_ac: "#FFFF00",
                scapula_aa: "#FFFF00", scapula_ts: "#FFFF00", scapula_ai: "#FFFF00"
            };

            Object.keys(lms).forEach(k => {
                const key = k as keyof typeof lms;
                const label = lms[key];
                const marker = markers[key];
                if (label && marker) {
                    label.visible = isLabelsEnabled.value;
                    if (!label.visible) return;
                    const worldPos = new THREE.Vector3();
                    marker.getWorldPosition(worldPos);
                    label.position.copy(worldPos).add(new THREE.Vector3(0, 10, 0));
                    const canvas = (label.material.map as THREE.CanvasTexture).image;
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                        ctx.clearRect(0, 0, 1024, 256);
                        ctx.fillStyle = 'rgba(0,0,0,0.7)';
                        ctx.fillRect(0, 0, 1024, 256);
                        ctx.font = 'bold 72px Inter, sans-serif';
                        ctx.fillStyle = colors[key];
                        const text = `${jointNames[key]} (${boneNames[key]})`;
                        ctx.fillText(text, 40, 160);
                        label.material.map!.needsUpdate = true;
                    }
                }
            });
        });
    };

    const updateKinematicChain = (side: 'right' | 'left') => {
      const cMesh = clavicleMeshes[side];
      const sMesh = scapulaMeshes[side];
      const hMesh = humerusMeshes[side];
      if (!cMesh || !sMesh || !hMesh) return;

      const coords = side === 'right' ? r_joint_coords.value : l_joint_coords.value;
      const pivots = jointPivots[side];
      
      // 1. Reset all bones to Zero-Pose (The ISB-aligned mean mesh)
      const allBones = [cMesh, sMesh, hMesh];
      allBones.forEach(b => {
        b.quaternion.set(0, 0, 0, 1); // Identity
        b.position.set(0,0,0); // Relative to world 0
      });

      // 2. Define Joint Rotation Quaternions (ISB Standards)
      // Per-side mirror so a positive slider produces the same anatomical
      // motion on both shoulders; defaults are symmetric L/R.
      const jointSide = side === 'right' ? 1 : -1;

      const qSC = new THREE.Quaternion().setFromEuler(new THREE.Euler(
        THREE.MathUtils.degToRad(-jointSide * coords.sc_elevation),  // X (elevation/depression)
        THREE.MathUtils.degToRad( jointSide * coords.sc_abduction),  // Y (protraction/retraction)
        THREE.MathUtils.degToRad(-jointSide * coords.sc_upward),     // Z (axial rotation)
        'YXZ'
      ));

      const qAC = new THREE.Quaternion().setFromEuler(new THREE.Euler(
        THREE.MathUtils.degToRad(-jointSide * coords.ac_upward),     // X (upward rotation)
        THREE.MathUtils.degToRad(-jointSide * coords.ac_internal),   // Y (internal/external rotation)
        THREE.MathUtils.degToRad(-jointSide * coords.ac_posterior),  // Z (posterior tilt)
        'YXZ'
      ));

      // GH rotates about anatomical axes (humerus hangs along -Y at neutral;
      // world X=anterior, Y=superior, Z=lateral):
      //   flexion   → mediolateral (Z)      forward swing, same sign both arms
      //   abduction → anteroposterior (X)    lateral raise, mirrored per side
      //   internal  → humeral long axis (Y)  axial spin, mirrored per side
      const qGH = new THREE.Quaternion().setFromEuler(new THREE.Euler(
        THREE.MathUtils.degToRad(jointSide * coords.gh_abduction),   // X
        THREE.MathUtils.degToRad(-jointSide * coords.gh_internal),   // Y
        THREE.MathUtils.degToRad(coords.gh_flexion),                // Z
        'YXZ'
      ));

      // 3. Recursive Transform Application
      
      // A. SC JOINT (Thorax vs Clavicle)
      cMesh.position.sub(pivots.sc);
      cMesh.position.applyQuaternion(qSC);
      cMesh.position.add(pivots.sc);
      cMesh.quaternion.premultiply(qSC);

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

      // D. GHOST SYNC (If overlap enabled, apply same logic to ghost meshes)
      if (isOverlapEnabled.value && meanModelData) {
          const g_c = side === 'right' ? ghostMeshes.clavicle.right : ghostMeshes.clavicle.left;
          const g_s = side === 'right' ? ghostMeshes.scapula.right : ghostMeshes.scapula.left;
          const g_h = side === 'right' ? ghostMeshes.humerus.right : ghostMeshes.humerus.left;
          if (g_c && g_s && g_h) {
              const g_pivots = meanModelData.isb_joints[side];
              const g_p = { sc: new THREE.Vector3(g_pivots.sc[0], g_pivots.sc[1], g_pivots.sc[2]), ac: new THREE.Vector3(g_pivots.ac[0], g_pivots.ac[1], g_pivots.ac[2]), gh: new THREE.Vector3(g_pivots.gh[0], g_pivots.gh[1], g_pivots.gh[2]) };
              
              [g_c, g_s, g_h].forEach(m => { m.quaternion.set(0,0,0,1); m.position.set(0,0,0); });

              g_c.position.sub(g_p.sc).applyQuaternion(qSC).add(g_p.sc);
              g_c.quaternion.premultiply(qSC);

              const gac_W = g_p.ac.clone().sub(g_p.sc).applyQuaternion(qSC).add(g_p.sc);
              g_s.position.sub(g_p.sc).applyQuaternion(qSC).add(g_p.sc);
              g_s.quaternion.premultiply(qSC);
              g_s.position.sub(gac_W).applyQuaternion(qAC).add(gac_W);
              g_s.quaternion.premultiply(qAC);

              const ggh_W = g_p.gh.clone().sub(g_p.ac).applyQuaternion(qAC).add(g_p.ac).sub(g_p.sc).applyQuaternion(qSC).add(g_p.sc);
              g_h.position.sub(g_p.sc).applyQuaternion(qSC).add(g_p.sc);
              g_h.quaternion.premultiply(qSC);
              g_h.position.sub(gac_W).applyQuaternion(qAC).add(gac_W);
              g_h.quaternion.premultiply(qAC);
              g_h.position.sub(ggh_W).applyQuaternion(qGH).add(ggh_W);
              g_h.quaternion.premultiply(qGH);
          }
      }
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

    // Ctrl+Shift+D toggles developer mode (persisted across reloads).
    window.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'D' || e.key === 'd')) {
        e.preventDefault();
        isDevMode.value = !isDevMode.value;
        localStorage.setItem('ssm_dev_mode', isDevMode.value ? '1' : '0');
        statusMessage.value = `Dev mode ${isDevMode.value ? 'ENABLED' : 'disabled'}`;
        statusColor.value = isDevMode.value ? '#FFA040' : '#94a3b8';
      }
    });
  }
});

async function runPrediction() {
  isPredicting.value = true;
  predictionProgress.value = 0;
  statusMessage.value = "Starting Python pipeline...";
  statusColor.value = "#ffffff";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        sex: sex.value,
        age: age.value,
        height: height.value,
        weight: weight.value,
        r_clav_len: r_clav_len.value,
        r_hum_len: r_hum_len.value,
        r_hum_epi_width: r_hum_epi_width.value,
        fabrik_step: 4
      })
    });

    if (!response.ok) throw await response.text();
    const boneData = await response.json();
    predictionProgress.value = 100;
    statusMessage.value = "Prediction Complete! Rendering...";
    statusColor.value = "#48c774";
    hasPrediction.value = true;
    isViewingOriginal.value = false;
    loadBones(boneData);
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
    const response = await fetch("/api/save_report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        patient: {
            sex: sex.value,
            age: age.value,
            height: height.value,
            weight: weight.value,
        },
        right_st: r_joint_coords.value,
        left_st: l_joint_coords.value,
      })
    });
    if (!response.ok) throw await response.text();
    statusMessage.value = await response.text();
    statusColor.value = "#48c774";
  } catch (error) {
    statusMessage.value = "Export Failed: " + error;
    statusColor.value = "#f14668";
  }
  isSavingReport.value = false;
}

async function runFabrikStep() {
  if (isPredicting.value) return;

  isPredicting.value = true;
  predictionProgress.value = 0;
  statusMessage.value = "Running FABRIK...";
  statusColor.value = "#FFA040";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        sex: sex.value,
        age: age.value,
        height: height.value,
        weight: weight.value,
        r_clav_len: r_clav_len.value,
        r_hum_len: r_hum_len.value,
        r_hum_epi_width: r_hum_epi_width.value,
        fabrik_step: 4
      })
    });

    if (!response.ok) throw await response.text();
    const boneData = await response.json();
    predictionProgress.value = 100;
    statusMessage.value = "FABRIK Complete!";
    statusColor.value = "#48c774";
    hasPrediction.value = true;
    isViewingOriginal.value = false;
    loadBones(boneData);
  } catch (error) {
    statusMessage.value = "FABRIK Failed: " + error;
    statusColor.value = "#f14668";
  }

  isPredicting.value = false;
}

function toggleComparison() {
  isViewingOriginal.value = !isViewingOriginal.value;
  loadBones();
}
</script>

<template>
  <div class="container" :class="{ 'light-mode': isLightMode }">
     <div class="left-pane">
        <div class="viewer-wrapper">
           <div class="floating-frame" ref="viewerContainer">
             <!-- Three.js Canvas -->
           </div>
           
           <!-- Viewport Overlay Label -->
           <div class="viewport-label animate-in">
              <div class="status-indicator" :class="{ active: !isViewingOriginal }"></div>
              <span class="label-text">
                {{ isViewingOriginal ? 'Mean Anatomical Model' : (hasPrediction ? 'Predicted Patient-Specific Mesh' : 'Initial Model') }}
              </span>
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
                <button @click="toggleTheme" class="icon-btn" :title="isLightMode ? 'Switch to dark mode' : 'Switch to light mode'">
                    <svg v-if="isLightMode" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
                    <svg v-else xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
                </button>
                <button v-if="hasPrediction" @click="toggleComparison" class="comparison-btn header-compare" :class="{ original: isViewingOriginal }">
                   <span v-if="isViewingOriginal">🔄 View Predicted Mesh</span>
                   <span v-else>📏 Compare with Mean Model</span>
                </button>
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

            <div v-else-if="isKinematicVisible" class="settings-view animate-in kinematic-scroll">
              <div class="card transparent-card">
                <h3 style="color: #FFA040">🦴 Kinematic Refinement</h3>
                <p class="hint">Adjust joint coordinates along the recursive chain.</p>
                
                <div class="joint-group">
                  <div class="side-label">Right Shoulder</div>
                  
                  <div class="sub-group">
                    <div class="group-title">Sternoclavicular (SC)</div>
                    <div class="slider-row">
                      <label><span>Abduction</span> <span>{{ r_joint_coords.sc_abduction.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="r_joint_coords.sc_abduction" min="-90" max="90" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Elevation</span> <span>{{ r_joint_coords.sc_elevation.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="r_joint_coords.sc_elevation" min="-90" max="90" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Upward Rot</span> <span>{{ r_joint_coords.sc_upward.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="r_joint_coords.sc_upward" min="-90" max="90" step="0.5" />
                    </div>
                  </div>

                  <div class="sub-group">
                    <div class="group-title">Acromioclavicular (AC)</div>
                    <div class="slider-row">
                      <label><span>Internal Rot</span> <span>{{ r_joint_coords.ac_internal.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="r_joint_coords.ac_internal" min="-90" max="90" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Upward Rot</span> <span>{{ r_joint_coords.ac_upward.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="r_joint_coords.ac_upward" min="-90" max="90" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Posterior Tilt</span> <span>{{ r_joint_coords.ac_posterior.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="r_joint_coords.ac_posterior" min="-90" max="90" step="0.5" />
                    </div>
                  </div>

                  <div class="sub-group">
                    <div class="group-title">Glenohumeral (GH)</div>
                    <div class="slider-row">
                      <label><span>Flexion</span> <span>{{ r_joint_coords.gh_flexion.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="r_joint_coords.gh_flexion" min="-60" max="180" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Abduction</span> <span>{{ r_joint_coords.gh_abduction.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="r_joint_coords.gh_abduction" min="0" max="180" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Internal Rot</span> <span>{{ r_joint_coords.gh_internal.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="r_joint_coords.gh_internal" min="-90" max="90" step="0.5" />
                    </div>
                  </div>
                </div>

                <div class="joint-group" style="margin-top: 20px">
                  <div class="side-label">Left Shoulder</div>
                  
                  <div class="sub-group">
                    <div class="group-title">Sternoclavicular (SC)</div>
                    <div class="slider-row">
                      <label><span>Abduction</span> <span>{{ l_joint_coords.sc_abduction.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="l_joint_coords.sc_abduction" min="-90" max="90" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Elevation</span> <span>{{ l_joint_coords.sc_elevation.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="l_joint_coords.sc_elevation" min="-90" max="90" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Upward Rot</span> <span>{{ l_joint_coords.sc_upward.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="l_joint_coords.sc_upward" min="-90" max="90" step="0.5" />
                    </div>
                  </div>

                  <div class="sub-group">
                    <div class="group-title">Acromioclavicular (AC)</div>
                    <div class="slider-row">
                      <label><span>Internal Rot</span> <span>{{ l_joint_coords.ac_internal.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="l_joint_coords.ac_internal" min="-90" max="90" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Upward Rot</span> <span>{{ l_joint_coords.ac_upward.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="l_joint_coords.ac_upward" min="-90" max="90" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Posterior Tilt</span> <span>{{ l_joint_coords.ac_posterior.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="l_joint_coords.ac_posterior" min="-90" max="90" step="0.5" />
                    </div>
                  </div>

                  <div class="sub-group">
                    <div class="group-title">Glenohumeral (GH)</div>
                    <div class="slider-row">
                      <label><span>Flexion</span> <span>{{ l_joint_coords.gh_flexion.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="l_joint_coords.gh_flexion" min="-60" max="180" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Abduction</span> <span>{{ l_joint_coords.gh_abduction.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="l_joint_coords.gh_abduction" min="0" max="180" step="0.5" />
                    </div>
                    <div class="slider-row">
                      <label><span>Internal Rot</span> <span>{{ l_joint_coords.gh_internal.toFixed(1) }}°</span></label>
                      <input type="range" v-model.number="l_joint_coords.gh_internal" min="-90" max="90" step="0.5" />
                    </div>
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
                <h3>🩺 Measurements</h3>
                <div class="grid-compact">
                  <div>
                    <label>Sex</label>
                    <select v-model="sex" class="input-fi">
                      <option value="0">Male</option>
                      <option value="1">Female</option>
                    </select>
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
                    <label>R Clavicle Length (mm)</label>
                    <input v-model="r_clav_len" class="input-fi" />
                  </div>
                  <div>
                    <label>R Humerus Length (mm)</label>
                    <input v-model="r_hum_len" class="input-fi" />
                  </div>
                  <div>
                    <label>R Hum Epicondyle Width (mm)</label>
                    <input v-model="r_hum_epi_width" class="input-fi" />
                  </div>
                </div>
              </div>

              <div class="card transparent-card" style="margin-top: 15px; border-color: #60A060;">
                <h3 style="color: #60A060">🔍 Anatomical Guides</h3>
                <div class="toggle-group" style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.08); color: white; display: flex; align-items: center; gap: 10px; font-weight: bold;">
                  <input type="checkbox" v-model="showGuides" @change="loadBones()" id="guidesToggle" />
                  <label for="guidesToggle">Show Visual Guides</label>
                </div>
                <div class="toggle-group" style="margin-bottom: 8px; color: white; display: flex; align-items: center; gap: 10px;" :style="{ opacity: showGuides ? 1 : 0.4 }">
                  <input type="checkbox" v-model="isHighlightsEnabled" :disabled="!showGuides" @change="loadBones()" id="highlightToggle" />
                  <label for="highlightToggle">Highlight Glide Area</label>
                </div>
                <div class="toggle-group" style="color: white; display: flex; align-items: center; gap: 10px;" :style="{ opacity: showGuides ? 1 : 0.4 }">
                  <input type="checkbox" v-model="isNormalsEnabled" :disabled="!showGuides" @change="loadBones()" id="normalsToggle" />
                  <label for="normalsToggle">Show Surface Normals</label>
                </div>
                <div class="toggle-group" style="margin-top: 5px; color: white; display: flex; align-items: center; gap: 10px;" :style="{ opacity: showGuides ? 1 : 0.4 }">
                  <input type="checkbox" v-model="isScapularPlaneEnabled" :disabled="!showGuides" @change="loadBones()" id="scapularPlaneToggle" />
                  <label for="scapularPlaneToggle">Show Scapular Plane</label>
                </div>
                <div class="toggle-group" style="margin-top: 5px; color: white; display: flex; align-items: center; gap: 10px;" :style="{ opacity: showGuides ? 1 : 0.4 }">
                  <input type="checkbox" v-model="isLabelsEnabled" :disabled="!showGuides" id="labelsToggle" />
                  <label for="labelsToggle">Show Coordinate Labels</label>
                </div>
              </div>

              <button :disabled="isPredicting" @click="runPrediction" class="run-btn" style="margin-top: 15px;">
                <span v-if="!isPredicting">🚀 Run Prediction Pipeline</span>
                <span v-else>🔄 Executing Model Generation...</span>
              </button>

              <button v-if="isDevMode" :disabled="isPredicting" @click="runFabrikStep" class="run-btn step-btn">
                <span v-if="!isPredicting">🛠️ Run FABRIK</span>
                <span v-else>⏳ Running...</span>
              </button>

              <div v-if="isPredicting" class="progress-wrap animate-in">
                <div class="progress-track">
                  <div class="progress-fill" :style="{ width: predictionProgress + '%' }"></div>
                </div>
                <div class="progress-pct">{{ Math.round(predictionProgress) }}%</div>
              </div>

              <div v-if="isDevMode && statusMessage" class="status-box" :style="{ color: statusColor, borderColor: statusColor }">
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
*, *::before, *::after {
  box-sizing: border-box;
}
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
  gap: 14px;
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}
.main-view, .settings-view {
  padding: 15px 25px;
  overflow-y: auto; /* Allow scrolling */
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
.step-btn {
  margin-top: 10px;
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  box-shadow: 0 10px 20px rgba(217, 119, 6, 0.2);
}
.step-btn:hover:not(:disabled) {
  box-shadow: 0 12px 24px rgba(217, 119, 6, 0.3);
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

/* Prediction progress bar */
.progress-wrap {
  margin-top: 15px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.progress-track {
  flex: 1;
  height: 10px;
  background-color: #161625;
  border: 1px solid rgba(255, 255, 255, 0.15);
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4facfe, #48c774);
  transition: width 0.4s ease;
}
.progress-pct {
  font-family: 'Consolas', monospace;
  font-size: 0.8rem;
  color: #cbd5e1;
  min-width: 3.5ch;
  text-align: right;
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

.sub-group {
  margin-top: 15px;
  border-top: 1px solid rgba(255,255,255,0.05);
  padding-top: 10px;
}
.group-title {
  font-size: 0.7rem;
  color: #FFA040;
  text-transform: uppercase;
  margin-bottom: 8px;
  font-weight: bold;
}
.kinematic-scroll {
  max-height: calc(100vh - 120px);
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

.viewport-label {
  position: absolute;
  top: 30px;
  left: 30px;
  padding: 10px 20px;
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
  z-index: 10;
  pointer-events: none;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.label-text {
  color: #f8fafc;
  font-size: 0.9rem;
  font-weight: 600;
  letter-spacing: 0.05rem;
  text-transform: uppercase;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #94a3b8;
  box-shadow: 0 0 10px rgba(148, 163, 184, 0.4);
  transition: all 0.3s ease;
}

.status-indicator.active {
  background: #3b82f6;
  box-shadow: 0 0 15px rgba(59, 130, 246, 0.8);
}

.comparison-toggle {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid rgba(255,255,255,0.05);
}
.pane-header h2 {
  white-space: nowrap;
}
/* Compact variant of .comparison-btn for the card header */
.header-compare {
  width: auto;
  padding: 7px 12px;
  font-size: 0.8rem;
  white-space: nowrap;
}
.comparison-btn {
  width: 100%;
  padding: 12px;
  background: rgba(255, 160, 64, 0.1);
  border: 1px solid rgba(255, 160, 64, 0.3);
  color: #FFA040;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.2s;
}
.comparison-btn:hover {
  background: rgba(255, 160, 64, 0.2);
}
.comparison-btn.original {
  background: rgba(72, 199, 116, 0.1);
  border-color: rgba(72, 199, 116, 0.3);
  color: #48c774;
}
.comparison-btn.original:hover {
  background: rgba(72, 199, 116, 0.2);
}

/* =========================================================================
   LIGHT MODE
   Overrides scoped under .container.light-mode. Light is the default theme;
   the header sun/moon button toggles it (persisted in localStorage).
   ========================================================================= */
.container.light-mode {
  background: radial-gradient(circle at 50% 50%, #f4f6fb 0%, #dbe2ee 100%);
  color: #1e293b;
}
.light-mode .floating-frame {
  background: rgba(255, 255, 255, 0.75);
  border: 1px solid rgba(0, 0, 0, 0.08);
  box-shadow: 0 30px 80px -20px rgba(30, 41, 59, 0.25),
              inset 0 0 20px rgba(255, 255, 255, 0.4);
}
.light-mode .frame-reflection {
  background: linear-gradient(135deg, rgba(255,255,255,0.5) 0%, transparent 40%);
}
.light-mode .pane-header {
  background: rgba(255, 255, 255, 0.6);
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}
.light-mode h2 { color: #0f172a; }
.light-mode .card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
}
/* Keep transparent cards borderless in light mode (the .card override above
   would otherwise re-add a border/background to them). */
.light-mode .transparent-card {
  background: transparent;
  border: none;
}
.light-mode .hint { color: #64748b; }
.light-mode label { color: #64748b; }
.light-mode .input-fi {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  color: #1e293b;
}
.light-mode .icon-btn {
  background: #e8ecf4;
  color: #475569;
}
.light-mode .icon-btn:hover {
  background: #d5dce8;
  color: #0f172a;
}
.light-mode .secondary-btn {
  background: #e8ecf4;
  color: #1e293b;
  border: 1px solid #cbd5e1;
}
.light-mode .secondary-btn:hover { background: #d5dce8; }
.light-mode .status-box { background-color: #f1f5f9; }
.light-mode .progress-track {
  background-color: #e2e8f0;
  border: 1px solid rgba(0, 0, 0, 0.12);
}
.light-mode .progress-pct { color: #475569; }
.light-mode .joint-group {
  background: rgba(15, 23, 42, 0.03);
  border: 1px solid rgba(15, 23, 42, 0.08);
}
.light-mode .side-label { color: #0f172a; }
.light-mode .sub-group { border-top: 1px solid rgba(15, 23, 42, 0.08); }
.light-mode .comparison-toggle { border-top: 1px solid rgba(15, 23, 42, 0.08); }
.light-mode .viewport-label {
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(0, 0, 0, 0.08);
  box-shadow: 0 8px 32px rgba(30, 41, 59, 0.15);
}
.light-mode .label-text { color: #0f172a; }
</style>
.overlap-btn {
  background: rgba(79, 172, 254, 0.1) !important;
  border-color: rgba(79, 172, 254, 0.3) !important;
  color: #70c5ff !important;
  margin-top: 8px;
  width: 100%;
}

.overlap-btn.active {
  background: rgba(79, 172, 254, 0.3) !important;
  border-color: #4facfe !important;
  box-shadow: 0 0 15px rgba(79, 172, 254, 0.4);
}
