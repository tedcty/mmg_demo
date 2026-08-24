<script setup lang="ts">
import { ref, computed, onMounted, watch } from "vue";
import * as THREE from 'three';

// Unique ID for this browser tab — isolates predictions and progress from other sessions.
// crypto.randomUUID() only exists in a secure context (HTTPS/localhost); fall back so
// plain-HTTP tablets still get a unique id instead of all colliding on one session.
const sessionId = (crypto?.randomUUID?.())
  || `s-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ViewHelper } from 'three/examples/jsm/helpers/ViewHelper.js';

// Paths
const anthro_path = ref("/home/trix/Dev/Repo/mmg_demo/mmg_demo/Demos/SSM Demo/predict_gui/Resources/anthro_data.csv");
const ssm_path = ref("/home/trix/Dev/Repo/mmg_demo/mmg_demo/Demos/SSM Demo/predict_gui/Resources/SSM_shape_model_103");
const out_path = ref("/home/trix/Dev/Repo/mmg_demo/mmg_demo/Demos/SSM Demo/predict_gui/Resources/predicted_model.ply");

// Patient Data — defaults to a typical adult female (~163 cm) so the demo opens
// with a realistic example rather than blank fields.
const sex = ref("1");            // Female
const age = ref("40");           // years
const height = ref("163");       // cm
const weight = ref("65");        // kg (~BMI 24.5)
const r_clav_len = ref("148");   // mm — right clavicle length
const r_hum_len = ref("305");    // mm — right humerus length
const r_hum_epi_width = ref("57"); // mm — right humeral epicondyle width

// Which bone measurement's diagram to highlight — driven by which field is
// focused, so the "How to Measure" guide shows where to take that measurement.
const activeMeasure = ref<"clav" | "hum" | "epi">("clav");
const MEASURE_INFO = {
  clav: {
    name: "Clavicle length",
    desc: "From the sternoclavicular joint (base of the throat) out to the acromioclavicular joint (bony tip of the shoulder), following the collarbone.",
  },
  hum: {
    name: "Humerus length",
    desc: "From the tip of the shoulder (acromion) down to the lateral epicondyle (bony bump on the outside of the elbow), arm relaxed at the side.",
  },
  epi: {
    name: "Epicondyle width",
    desc: "Straight-line distance across the elbow between the two bony bumps — the medial and lateral epicondyles — with the elbow bent 90°.",
  },
} as const;
const measureInfo = computed(() => MEASURE_INFO[activeMeasure.value]);

// Pre-labelled upper-body skeleton illustrations (in Documents/resources, served
// by the demo server). One image per measurement, with the relevant bone
// highlighted. The male-specific images fall back to the female illustration
// until they exist (see onGuideError).
const MEASURE_IMG = { clav: "clavicle", hum: "humerus", epi: "epicondyles" } as const;
const guideImage = computed(() => {
  const who = sex.value === "0" ? "male" : "female";
  return `/doc-resources/${who}_upperbody_${MEASURE_IMG[activeMeasure.value]}_cropped.png`;
});
function onGuideError(e: Event) {
  const img = e.target as HTMLImageElement;
  const fallback = `/doc-resources/female_upperbody_${MEASURE_IMG[activeMeasure.value]}_cropped.png`;
  if (!img.src.endsWith(fallback)) img.src = fallback;
}

// The guide is only shown while one of the bone-measurement fields is focused.
// A short blur delay avoids flicker when tabbing between the three fields.
const showMeasureGuide = ref(false);
// The guide card slots directly below the focused field via CSS order.
const guideOrder = computed(() => ({ clav: 2, hum: 4, epi: 6 })[activeMeasure.value]);

// iOS/Android overlay the on-screen keyboard without shrinking innerHeight, so
// visualViewport.height is the only reliable signal for the space that's
// actually visible. Track it to (a) know when the keyboard is up and (b) cap the
// guide image so the field + guide fit above the keyboard.
const vvHeight = ref(typeof window !== "undefined" ? window.innerHeight : 0);
const keyboardUp = computed(() => window.innerHeight - vvHeight.value > 120);
const guideImgStyle = computed(() =>
  keyboardUp.value ? { maxHeight: `${Math.max(150, Math.round(vvHeight.value - 250))}px` } : {}
);
onMounted(() => {
  const vv = window.visualViewport;
  if (!vv) return;
  const update = () => { vvHeight.value = vv.height; };
  vv.addEventListener("resize", update);
  vv.addEventListener("scroll", update);
  update();
});

let measureBlurTimer: ReturnType<typeof setTimeout> | undefined;
function onMeasureFocus(m: "clav" | "hum" | "epi", e?: FocusEvent) {
  if (measureBlurTimer) clearTimeout(measureBlurTimer);
  activeMeasure.value = m;
  showMeasureGuide.value = true;
  // Scroll the field to the top of the panel *once the keyboard has opened*
  // (visualViewport shrinks), so the field and the guide beneath it clear it.
  const field = (e?.target as HTMLElement | undefined)?.closest(".bf");
  if (!field) return;
  const vv = window.visualViewport;
  if (!vv) { setTimeout(() => field.scrollIntoView({ block: "start", behavior: "smooth" }), 350); return; }
  const startH = vv.height;
  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    vv.removeEventListener("resize", onKbResize);
    field.scrollIntoView({ block: "start", behavior: "smooth" });
  };
  const onKbResize = () => { if (vv.height < startH - 100) finish(); };
  vv.addEventListener("resize", onKbResize);
  setTimeout(finish, 700);   // fallback: no keyboard (desktop) or slow to open
}
function onMeasureBlur() {
  measureBlurTimer = setTimeout(() => { showMeasureGuide.value = false; }, 120);
}

const statusMessage = ref("");
const statusColor = ref("#ffffff");
const isPredicting = ref(false);
// Prediction progress (0–100), driven by the pipeline's streamed stage
// messages. Shown to all users as the demo-facing feedback while the raw
// "Pipeline Output" text stays dev-only.
const predictionProgress = ref(0);
// Timer that eases the bar forward during long silent gaps (e.g. the heavy
// Python imports emit no output). Real stage messages still jump it via Math.max.
let progressTimer: ReturnType<typeof setInterval> | undefined;

// Ordered pipeline stages → target progress %. Matched by substring against
// the STATUS messages streamed from the Python pipeline so the bar advances
// as each stage begins. Progress is kept monotonic (only ever increases), so
// stages that get skipped (e.g. cached PLSR training) don't stall the bar.
const PIPELINE_STAGES: { match: string; pct: number }[] = [
  { match: "Initialising",       pct: 5 },
  { match: "Loading libraries",  pct: 10 },
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
// Kinematics tab is kept fully working but hidden from the toolbar while the
// PC (Shape) tab is being debugged — flip this back to true to bring it back.
const SHOW_KINEMATICS_TAB = false;
const isKinematicVisible = ref(false);
const isPcVisible = ref(false);
// PC (Shape) adjustment — sliders over the shape model's principal-component
// weights, so PC changes can be previewed independently of the anthropometric
// PLSR prediction. Populated from /api/pc_info in the background as soon as
// the app mounts (see onMounted's fetchPcInfo() call), so by the time the
// user opens the tab the one-time server-side model load is likely already
// done rather than felt as a UI delay.
const pcInfo = ref<{ n_modes: number; std: number[]; variance_pct: number[] } | null>(null);
// Sliders are expressed in standard deviations (not raw PC weights) — much
// easier to read ("1.5 SD from the mean") — and converted to raw weights
// (sd * std[i]) only when sent to the backend.
const pcSd = ref<number[]>([]);
const PC_SD_RANGE = 2;
const isPcLoading = ref(false);
const isPcUpdating = ref(false);
const isPcInfoOpen = ref(false); // "About this tab" popup, toggled by the info-icon button

// Client-side live-preview data (see fetchPcClientData/applyPcLiveFromSd
// below) — the frozen per-bone rigid-transform recipe plus the raw
// mean-mesh + PC-mode vertex arrays needed to reconstruct and skin the mesh
// entirely in the browser, so slider *dragging* is instant with no network
// round trip. Held as plain module-level variables (not ref()s) since these
// arrays are large (~16MB) and never need Vue's reactivity — only read
// inside applyPcLiveFromSd, never rendered.
interface PcBoneMetaCommon {
  validIds: number[];
  indices: number[];
  origin: [number, number, number];
  ij: [number, number, number];
  mat: number[]; // 16 floats, column-major — matches THREE.Matrix4.fromArray
}
// Vertex-id lists used to derive fresh joint centers each frame (see
// computeLiveJoints below) — the same landmark/sphere-fit CSVs
// bones/base_bone.py's get_landmark/get_sphere_center read server-side.
interface PcThoraxLandmarks { scRSphereIds: number[]; scLSphereIds: number[]; }
interface PcClavicleLandmarks { acIds: number[]; scSphereIds: number[]; }
interface PcScapulaLandmarks { aaIds: number[]; tsIds: number[]; aiIds: number[]; cpIds: number[]; ghSphereIds: number[]; }
interface PcHumerusLandmarks { ghSphereIds: number[]; }

interface PcThoraxMeta extends PcBoneMetaCommon { kind: 'thorax'; landmarks: PcThoraxLandmarks; }
interface PcClavicleMeta extends PcBoneMetaCommon {
  kind: 'clavicle';
  offset: [number, number, number];
  // Frozen fallback values (see computeLiveJoints, which recomputes these
  // fresh from `landmarks` every frame instead of using them directly) —
  // sync_to_scapula's rotation (snaps the AC end onto the scapula's solved
  // AC joint), applied around scJoint after the base seed transform below.
  scJoint: [number, number, number];
  syncQuat: [number, number, number, number];
  landmarks: PcClavicleLandmarks;
}
interface PcHumerusMeta extends PcBoneMetaCommon { kind: 'humerus'; offset: [number, number, number]; landmarks: PcHumerusLandmarks; }
interface PcScapulaMeta extends PcBoneMetaCommon {
  kind: 'scapula';
  offset: [number, number, number];
  seed: [number, number, number];
  quat: [number, number, number, number];
  solvedPos: [number, number, number];
  landmarks: PcScapulaLandmarks;
}
type PcBoneMeta = PcThoraxMeta | PcClavicleMeta | PcHumerusMeta | PcScapulaMeta;
interface PcClientMeta {
  nVerts: number;
  nModes: number;
  pcStd: number[];
  bones: {
    thorax: PcThoraxMeta;
    clav_r: PcClavicleMeta; clav_l: PcClavicleMeta;
    scap_r: PcScapulaMeta; scap_l: PcScapulaMeta;
    hum_r: PcHumerusMeta; hum_l: PcHumerusMeta;
  };
}
let pcClientMeta: PcClientMeta | null = null;
let pcClientMeanVerts: Float32Array | null = null;
let pcClientModes: Float32Array | null = null; // n_modes blocks of n_verts*3 floats, mode-major
let pcClientLoading = false;
// Set by onPcSliderInput, consumed once per rendered frame by the animate()
// loop — see the comment there for why this indirection exists (coalescing
// many 'input' events per frame down to one recompute).
let pcLiveDirty = false;
// Side panel (Shoulder Predictor) — hidden by default so the 3D viewport is
// full-screen; toggled open with the hamburger button.
const isPanelOpen = ref(false);
// Dev mode — gates developer-only tools (e.g. Run FABRIK). Toggle with
// Ctrl+Shift+D; persisted so it survives reloads. Hidden from demo users.
const isDevMode = ref(localStorage.getItem('ssm_dev_mode') === '1');
// Colour theme — light mode is the default. Persisted so the choice survives
// reloads; only an explicit 'dark' opts out of light.
const isLightMode = ref(localStorage.getItem('ssm_theme') !== 'dark');
const SCENE_BG = { light: 0xe8ecf4, dark: 0x0c0c48 };

// Off (default) = fast prediction: reuses the mean model's joint
// orientation, only joint *positions* are re-derived from this person's
// anatomy (see server.py's _predict_fast). On = the original full per-person
// FABRIK solve — slower but derives this person's own joint orientation
// too. Persisted so the choice survives reloads.
const useFullJointSolve = ref(localStorage.getItem('ssm_full_joint_solve') === '1');
watch(useFullJointSolve, (v) => localStorage.setItem('ssm_full_joint_solve', v ? '1' : '0'));

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
const gizmoCanvas = ref<HTMLCanvasElement | null>(null);
let globalScene: THREE.Scene | null = null;
let globalCamera: THREE.PerspectiveCamera | null = null;
let globalControls: OrbitControls | null = null;
let bonesGroup: THREE.Group | null = null;
let ghostGroup: THREE.Group | null = null; // Group for the mean "ghost" overlap
let isFirstLoad = true;

// Comparison State
// Three independent model slots can exist at once — the mean model (always
// available), an anthropometric prediction, and a PC-weight shape
// adjustment — so the user can switch between whichever they've generated
// without one overwriting the other. viewMode picks which is currently shown.
type ViewMode = 'mean' | 'predicted' | 'pcShape';
const viewMode = ref<ViewMode>('mean');
const isOverlapEnabled = ref(false); // New: Overlap Mode state
const hasPrediction = ref(false); // an anthropometric prediction has been run
const hasPcShape = ref(false);    // a PC-weight shape adjustment has been made
// Viewport model picker — the label doubles as a dropdown to switch models.
const isModelListOpen = ref(false);
const showGuides = ref(false); // Master toggle: spheres, triangles, muscle/glide areas, labels
const isHighlightsEnabled = ref(true); // Control for glide area visualization
const isNormalsEnabled = ref(false); // Control for surface normals
const isScapularPlaneEnabled = ref(false); // Control for scapular plane
const isLabelsEnabled = ref(true); // Control for coordinate labels
const isMusclePointsEnabled = ref(false); // Control for subscapularis muscle point cloud — off by default (clutters the view)
let highlightsGroup: THREE.Group | null = null;
let scapularPlaneGroup: THREE.Group | null = null;
let meanModelData: any = null;
let predictedModelData: any = null;
let pcShapeModelData: any = null;

// Reference storage for ghost meshes
const ghostMeshes = {
    thorax: null as THREE.Mesh | null,
    clavicle: { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null },
    scapula: { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null },
    humerus: { right: null as THREE.Mesh | null, left: null as THREE.Mesh | null }
};

// `kind` says which slot a freshly-received `externalData` belongs to, so
// this can store an anthropometric prediction and a PC-shape adjustment
// independently instead of one overwriting the other. Omitted (or called
// with no externalData at all — e.g. a guide-toggle re-render) means "just
// re-render whatever viewMode currently points at from its existing data",
// falling back to a fresh /bones.json fetch only for the mean model.
async function loadBones(externalData: any = null, kind: 'predicted' | 'pcShape' | null = null) {
  if (!globalScene) return;

  try {
    let data;
    if (externalData) {
      data = externalData;
      console.log("Loading bones from injected Rust data...");
    } else if (viewMode.value === 'predicted' && predictedModelData) {
      data = predictedModelData;
    } else if (viewMode.value === 'pcShape' && pcShapeModelData) {
      data = pcShapeModelData;
    } else {
      // Add cache-busting timestamp to ensure we get the fresh bones.json
      const response = await fetch(`/bones.json?t=${Date.now()}`);
      data = await response.json();
      console.log("Loading bones from public/bones.json...");
    }

    if (!meanModelData) {
      meanModelData = JSON.parse(JSON.stringify(data));
    }
    if (externalData && kind === 'predicted') {
      predictedModelData = data;
      hasPrediction.value = true;
      viewMode.value = 'predicted';
    } else if (externalData && kind === 'pcShape') {
      pcShapeModelData = data;
      hasPcShape.value = true;
      viewMode.value = 'pcShape';
    }

    const activeData = viewMode.value === 'mean' ? meanModelData
      : viewMode.value === 'predicted' ? predictedModelData
      : pcShapeModelData;
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
              color: viewMode.value === 'mean' ? "#88aaff" : bone.color,
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
        else if (bone.label === "R Subscapularis") { subscapMeshes.right = mesh as THREE.Points; mesh.visible = isMusclePointsEnabled.value; }
        else if (bone.label === "L Subscapularis") { subscapMeshes.left = mesh as THREE.Points; mesh.visible = isMusclePointsEnabled.value; }
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
                  mesh.material.color.set(viewMode.value === 'mean' ? "#88aaff" : bone.color);
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
  // Pre-warm the Shape (PCA) tab's model in the background as soon as the
  // page loads, rather than waiting for the user to open that tab — by the
  // time they click it, the one-time server-side load (heavy library
  // imports + one reference assembly pass, a few seconds) is likely already
  // done or well underway. Fire-and-forget; fetchPcInfo already guards
  // against duplicate/overlapping calls.
  fetchPcInfo();
  // Also pre-warm the client-side live-preview data (~16MB one-time
  // download) so it's likely ready by the time the user starts dragging.
  fetchPcClientData();

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

    // Blender-style orientation gizmo, drawn in its own 128×128 canvas above the
    // toolbar. Its centre shares the OrbitControls target so axis snaps orbit the
    // model. Clicking an axis animates the main camera to that view.
    let viewHelper: ViewHelper | null = null;
    let gizmoRenderer: THREE.WebGLRenderer | null = null;
    if (gizmoCanvas.value) {
      gizmoRenderer = new THREE.WebGLRenderer({ canvas: gizmoCanvas.value, alpha: true, antialias: true });
      gizmoRenderer.setPixelRatio(window.devicePixelRatio);
      gizmoRenderer.setSize(128, 128);
      viewHelper = new ViewHelper(camera, gizmoCanvas.value);
      viewHelper.center = controls.target; // share reference so it tracks re-centring
      gizmoCanvas.value.addEventListener('click', (e) => {
        if (viewHelper) viewHelper.handleClick(e);
      });
    }

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

    const clock = new THREE.Clock();
    const animate = () => {
      requestAnimationFrame(animate);
      const delta = clock.getDelta();
      // Apply at most once per rendered frame (~60fps) rather than once per
      // 'input' DOM event — pointer input events can fire far faster than
      // the screen redraws, and each application recomputes vertex normals
      // across ~107k vertices (7 bones); doing that many times more often
      // than a frame can even show it was the actual cause of drag lag,
      // not the client-vs-backend split itself (see onPcSliderInput).
      if (pcLiveDirty) {
        pcLiveDirty = false;
        applyPcLiveFromSd(pcSd.value);
      }
      updateKinematicChain('right');
      updateKinematicChain('left');
      if (viewHelper && viewHelper.animating) viewHelper.update(delta);
      controls.update();
      renderer.render(scene, camera);
      // Gizmo overlays its own canvas; skip while hidden behind the panel.
      if (viewHelper && gizmoRenderer && !isPanelOpen.value) viewHelper.render(gizmoRenderer);
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
        statusColor.value = isDevMode.value ? '#00caef' : '#94a3b8';
      }
    });
  }
});

async function runPrediction() {
  isPredicting.value = true;
  predictionProgress.value = 0;
  statusMessage.value = "Starting Python pipeline...";
  statusColor.value = "#ffffff";

  // Ease toward 90% while waiting, decelerating as it goes, so the bar never
  // looks frozen during the long import phase. Stage messages override upward.
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = setInterval(() => {
    if (predictionProgress.value < 90) {
      predictionProgress.value += (90 - predictionProgress.value) * 0.02;
    }
  }, 500);

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
        fabrik_step: 4,
        use_full_solve: useFullJointSolve.value,
      })
    });

    if (!response.ok) throw await response.text();
    const boneData = await response.json();
    predictionProgress.value = 100;
    statusMessage.value = "Prediction Complete! Rendering...";
    statusColor.value = "#48c774";
    loadBones(boneData, 'predicted');
  } catch (error) {
    statusMessage.value = "Failed: " + error;
    statusColor.value = "#f14668";
  } finally {
    if (progressTimer) { clearInterval(progressTimer); progressTimer = undefined; }
  }

  isPredicting.value = false;
}

// Lazily loads PCA mode metadata (std dev + variance% per mode) the first
// time the Shape (PCA) tab is opened, then initializes one weight slider per
// mode at 0 (= mean shape). Cached in pcInfo so re-opening the tab is instant.
async function fetchPcInfo() {
  if (pcInfo.value || isPcLoading.value) return;
  isPcLoading.value = true;
  statusMessage.value = "Loading shape model (first time can take ~1-2 min)...";
  statusColor.value = "#ffffff";
  try {
    const response = await fetch("/api/pc_info");
    if (!response.ok) throw await response.text();
    pcInfo.value = await response.json();
    pcSd.value = new Array(pcInfo.value!.n_modes).fill(0);
    statusMessage.value = "Shape model loaded.";
    statusColor.value = "#48c774";
  } catch (error) {
    statusMessage.value = "Failed to load shape model: " + error;
    statusColor.value = "#f14668";
  }
  isPcLoading.value = false;
}

// One-time ~16MB fetch (mean mesh + 10 PC modes, float32) that makes fully
// client-side live mesh preview possible — see applyPcLiveFromSd below.
// Fire-and-forget background pre-warm, same pattern as fetchPcInfo; guarded
// against duplicate/overlapping calls. If this hasn't finished (or failed)
// by the time the user starts dragging, onPcSliderInput falls back to the
// original debounced-backend path, so there's no hard dependency on it.
async function fetchPcClientData() {
  if (pcClientMeta || pcClientLoading) return;
  pcClientLoading = true;
  try {
    const metaResp = await fetch("/api/pc_client_meta");
    if (!metaResp.ok) throw await metaResp.text();
    const meta = (await metaResp.json()) as PcClientMeta;

    const meshResp = await fetch("/api/pc_client_mesh");
    if (!meshResp.ok) throw await meshResp.text();
    const floats = new Float32Array(await meshResp.arrayBuffer());
    const n3 = meta.nVerts * 3;
    pcClientMeanVerts = floats.subarray(0, n3);
    pcClientModes = floats.subarray(n3);
    pcClientMeta = meta;
  } catch (error) {
    console.error("Failed to load PC client-side live-preview data (slider drag will fall back to the backend path):", error);
  }
  pcClientLoading = false;
}

// mean_verts + Σ weight_i · mode_i, reused across calls via one scratch
// buffer to avoid allocating ~1.5MB on every slider tick.
let _pcScratchVerts: Float32Array | null = null;
function reconstructPcVertsLive(weights: number[]): Float32Array | null {
  if (!pcClientMeta || !pcClientMeanVerts || !pcClientModes) return null;
  const n3 = pcClientMeta.nVerts * 3;
  if (!_pcScratchVerts || _pcScratchVerts.length !== n3) {
    _pcScratchVerts = new Float32Array(n3);
  }
  const combined = _pcScratchVerts;
  combined.set(pcClientMeanVerts);
  const nModes = Math.min(weights.length, pcClientMeta.nModes);
  for (let m = 0; m < nModes; m++) {
    const w = weights[m];
    if (!w) continue;
    const base = m * n3;
    for (let i = 0; i < n3; i++) {
      combined[i] += pcClientModes[base + i] * w;
    }
  }
  return combined;
}

// ── Live joint-center recompute ─────────────────────────────────────────
// Ports bones/base_bone.py's get_landmark/get_sphere_center plus each
// bone's replay() joint-derivation formula to TS, so joint centers (SC/AC/
// GH) are freshly re-derived from the live-reconstructed mesh every dirty
// frame instead of held at their frozen reference-shape value. Bone
// *orientation* (mat/ij, from PcBoneMetaCommon) stays frozen either way —
// only these joint *positions* move. See the "Recompute joint centers live
// during PC-slider drag" plan for the full derivation.
type Vec3 = [number, number, number];
function vSub(a: Vec3, b: Vec3): Vec3 { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function vAdd(a: Vec3, b: Vec3): Vec3 { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }

// Direct port of BoneBase.get_landmark: mean of the reconstructed vertices
// at `ids`.
function meanLandmark(combined: Float32Array, ids: number[]): Vec3 {
  let x = 0, y = 0, z = 0;
  for (const id of ids) {
    const s = id * 3;
    x += combined[s]; y += combined[s + 1]; z += combined[s + 2];
  }
  const n = ids.length;
  return [x / n, y / n, z / n];
}

// 3x3 linear solve via Cramer's rule — used by sphereFit below. Returns
// null (caller falls back to the landmark mean) if the system is singular,
// which real anatomical landmark sets should never hit.
function solve3x3(A: number[][], b: Vec3): Vec3 | null {
  const det3 = (m: number[][]) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const d = det3(A);
  if (Math.abs(d) < 1e-9) return null;
  const withCol = (col: number, v: Vec3): number[][] => {
    const m = [A[0].slice(), A[1].slice(), A[2].slice()];
    m[0][col] = v[0]; m[1][col] = v[1]; m[2][col] = v[2];
    return m;
  };
  return [det3(withCol(0, b)) / d, det3(withCol(1, b)) / d, det3(withCol(2, b)) / d];
}

// Direct, term-for-term port of BoneBase.sphere_fit (base_bone.py) — a
// least-squares sphere center through `ids`' reconstructed vertices. Kept
// structurally identical to the Python (not a reformulated/"optimized"
// version) so it stays byte-for-byte comparable when cross-checked against
// the backend.
function sphereFit(combined: Float32Array, ids: number[]): Vec3 {
  const n = ids.length;
  const mean = meanLandmark(combined, ids);
  const a: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const b: Vec3 = [0, 0, 0];
  for (const id of ids) {
    const s = id * 3;
    const p: Vec3 = [combined[s], combined[s + 1], combined[s + 2]];
    const d: Vec3 = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) a[i][j] += p[i] * d[j];
    const sq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    b[0] += sq * d[0]; b[1] += sq * d[1]; b[2] += sq * d[2];
  }
  for (let i = 0; i < 3; i++) { for (let j = 0; j < 3; j++) a[i][j] = (2 * a[i][j]) / n; b[i] /= n; }
  // Solve (AᵀA)c = Aᵀb, matching np.linalg.solve(a.T@a, a.T@b).
  const at = [[a[0][0], a[1][0], a[2][0]], [a[0][1], a[1][1], a[2][1]], [a[0][2], a[1][2], a[2][2]]];
  const ata: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let s = 0; for (let k = 0; k < 3; k++) s += at[i][k] * a[k][j]; ata[i][j] = s; }
  const atb: Vec3 = [0, 0, 0];
  for (let i = 0; i < 3; i++) { let s = 0; for (let k = 0; k < 3; k++) s += at[i][k] * b[k]; atb[i] = s; }
  return solve3x3(ata, atb) ?? mean;
}

const _pcRotMat = new THREE.Matrix4();
const _pcRotVec = new THREE.Vector3();
// mat * (p - ij) — a frozen bone's rotation-only transform, direct
// equivalent of transform_mesh(v, ij, mat) for a single point. mat's 4x4
// translation column is always ~0 for these bones (built from pure 3x3
// point-set alignments via Cloud.transform_between_3x3_points_sets), so
// applyMatrix4 (which includes translation) still matches exactly.
function applyFrozenRot(mat16: number[], ij: Vec3, p: Vec3): Vec3 {
  _pcRotMat.fromArray(mat16);
  _pcRotVec.set(p[0] - ij[0], p[1] - ij[1], p[2] - ij[2]);
  _pcRotVec.applyMatrix4(_pcRotMat);
  return [_pcRotVec.x, _pcRotVec.y, _pcRotVec.z];
}

const _pcPivotVec = new THREE.Vector3();
const _pcPivotQuat = new THREE.Quaternion();
// quat.apply(p - pivot) + target — the scapula FABRIK-correction /
// clavicle sync_to_scapula pivot-rotate stage, as a reusable single-point
// operation (also used to carry scapula's gh_joint_seed through its own
// pivot stage before humerus consumes it).
function pivotRotate(p: Vec3, pivot: Vec3, quat: [number, number, number, number], target: Vec3): Vec3 {
  _pcPivotVec.set(p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]);
  _pcPivotQuat.set(quat[0], quat[1], quat[2], quat[3]);
  _pcPivotVec.applyQuaternion(_pcPivotQuat);
  return [_pcPivotVec.x + target[0], _pcPivotVec.y + target[1], _pcPivotVec.z + target[2]];
}

const _pcSyncFrom = new THREE.Vector3();
const _pcSyncTo = new THREE.Vector3();
const _pcSyncQuatOut = new THREE.Quaternion();
// Direct equivalent of Clavicle.sync_to_scapula's
// scipy.spatial.transform.Rotation.align_vectors([v_new],[v_old]) — the
// minimal rotation aligning direction vOld to vNew — via THREE's built-in
// shortest-arc quaternion. Guards near-zero-length inputs the same way the
// Python does (identity rotation).
function syncRotation(vOld: Vec3, vNew: Vec3): [number, number, number, number] {
  const lenOld = Math.hypot(vOld[0], vOld[1], vOld[2]);
  const lenNew = Math.hypot(vNew[0], vNew[1], vNew[2]);
  if (lenOld < 1e-6 || lenNew < 1e-6) return [0, 0, 0, 1];
  _pcSyncFrom.set(vOld[0] / lenOld, vOld[1] / lenOld, vOld[2] / lenOld);
  _pcSyncTo.set(vNew[0] / lenNew, vNew[1] / lenNew, vNew[2] / lenNew);
  _pcSyncQuatOut.setFromUnitVectors(_pcSyncFrom, _pcSyncTo);
  return [_pcSyncQuatOut.x, _pcSyncQuatOut.y, _pcSyncQuatOut.z, _pcSyncQuatOut.w];
}

// Per-bone "how to skin this frame" recipe consumed by skinBoneLive, in
// addition to the always-frozen meta.ij/meta.mat. offset alone = a single
// translate after the base rotation (clavicle/scapula/humerus); offset +
// pivot/pivotTarget/quat = translate then a pivot-rotate stage on top
// (scapula's FABRIK correction, clavicle's sync_to_scapula). Thorax uses
// neither (mesh transform never depends on any joint center).
interface LiveBoneTransform {
  offset: Vec3 | null;
  pivot: Vec3 | null;
  pivotTarget: Vec3 | null;
  quat: [number, number, number, number] | null;
}
interface LiveSideJoints {
  clav: LiveBoneTransform;
  scap: LiveBoneTransform;
  hum: LiveBoneTransform;
  scJoint: Vec3;  // clavicle/thorax SC joint — also this side's clavicle origin marker
  acJoint: Vec3;  // scapula's solved AC joint — also this side's scapula origin marker
  ghJoint: Vec3;  // scapula's gh_joint_seed — also this side's humerus origin marker
}
interface LiveJoints { right: LiveSideJoints; left: LiveSideJoints; }

// Replicates one side of generate_isb_joints.replay_shape's call order —
// Clavicle (pre-sync) → Scapula (FABRIK delta reused, frozen quat) →
// Clavicle.sync_to_scapula (fresh quat) → Humerus — using only frozen
// orientation data (already in *Meta) plus landmarks freshly measured on
// `combined`.
function computeSideLiveJoints(
  combined: Float32Array,
  thoraxSc: Vec3,
  clavMeta: PcClavicleMeta,
  scapMeta: PcScapulaMeta,
  humMeta: PcHumerusMeta,
): LiveSideJoints {
  // --- Clavicle, pre-sync seed pose (Clavicle.replay, before sync_to_scapula) ---
  const acPt = meanLandmark(combined, clavMeta.landmarks.acIds);
  const scRaw = sphereFit(combined, clavMeta.landmarks.scSphereIds);
  const scWorld = thoraxSc;
  const clavOffset = vSub(scWorld, applyFrozenRot(clavMeta.mat, clavMeta.ij, scRaw));
  const acPresync = vAdd(applyFrozenRot(clavMeta.mat, clavMeta.ij, acPt), clavOffset);

  // --- Scapula (Scapula.replay) ---
  const aaRaw = meanLandmark(combined, scapMeta.landmarks.aaIds);
  const ghRawScap = sphereFit(combined, scapMeta.landmarks.ghSphereIds);
  const acSeed = acPresync; // clavicle's own pre-sync ac_joint feeds scapula as its seed anchor
  const scapOffset = vSub(acSeed, applyFrozenRot(scapMeta.mat, scapMeta.ij, aaRaw));
  const scapSeed = (pt: Vec3): Vec3 => vAdd(applyFrozenRot(scapMeta.mat, scapMeta.ij, pt), scapOffset);
  // Frozen FABRIK Step-4 delta (solvedPos - seed) — a constant, reused
  // exactly like Scapula.replay's `self.ac_joint - self._ac_seed`.
  const frozenDelta: Vec3 = vSub(scapMeta.solvedPos, scapMeta.seed);
  const acSol = vAdd(acSeed, frozenDelta);
  const ghJointSeed = pivotRotate(scapSeed(ghRawScap), acSeed, scapMeta.quat, acSol);

  // --- Clavicle sync_to_scapula (fresh rotation, runs after Scapula.replay) ---
  const syncQuat = syncRotation(vSub(acPresync, scWorld), vSub(acSol, scWorld));

  // --- Humerus (Humerus.replay) ---
  const ghRawHum = sphereFit(combined, humMeta.landmarks.ghSphereIds);
  const humOffset = vSub(ghJointSeed, applyFrozenRot(humMeta.mat, humMeta.ij, ghRawHum));

  return {
    clav: { offset: clavOffset, pivot: scWorld, pivotTarget: scWorld, quat: syncQuat },
    scap: { offset: scapOffset, pivot: acSeed, pivotTarget: acSol, quat: scapMeta.quat },
    hum:  { offset: humOffset, pivot: null, pivotTarget: null, quat: null },
    scJoint: scWorld,
    acJoint: acSol,
    ghJoint: ghJointSeed,
  };
}

// Top-level entry point, run once per dirty frame (same pcLiveDirty gate as
// the mesh skinning) — Thorax first (both SC joints), then each side.
function computeLiveJoints(combined: Float32Array): LiveJoints | null {
  if (!pcClientMeta) return null;
  const bones = pcClientMeta.bones;
  const thorax = bones.thorax;
  const scR = applyFrozenRot(thorax.mat, thorax.ij, sphereFit(combined, thorax.landmarks.scRSphereIds));
  const scL = applyFrozenRot(thorax.mat, thorax.ij, sphereFit(combined, thorax.landmarks.scLSphereIds));
  return {
    right: computeSideLiveJoints(combined, scR, bones.clav_r, bones.scap_r, bones.hum_r),
    left:  computeSideLiveJoints(combined, scL, bones.clav_l, bones.scap_l, bones.hum_l),
  };
}

// Reusable scratch objects for skinBoneLive — this runs over ~123k vertices
// per full slider update, so avoiding one allocation per vertex matters.
const _pcSkinMat = new THREE.Matrix4();
const _pcSkinQuat = new THREE.Quaternion();
const _pcSkinVec = new THREE.Vector3();

// Applies one bone's rigid-transform recipe for this frame to `combined`'s
// reconstructed vertices, writing straight into the already-live mesh's own
// position buffer. `meta.ij`/`meta.mat` (orientation) are always frozen;
// `live` (offset/pivot/pivotTarget/quat, from computeLiveJoints) carries
// this frame's freshly-derived joint-center data — null for thorax, whose
// mesh transform never depends on a joint center.
function skinBoneLive(meta: PcBoneMeta | undefined, live: LiveBoneTransform | null, combined: Float32Array, mesh: THREE.Mesh | null) {
  if (!meta || !mesh) return;
  const posAttr = mesh.geometry.getAttribute("position") as THREE.BufferAttribute | undefined;
  const ids = meta.validIds;
  if (!posAttr || posAttr.count !== ids.length) return;

  _pcSkinMat.fromArray(meta.mat);
  const [ijx, ijy, ijz] = meta.ij;
  const offset = live?.offset ?? null;
  const pivot = live?.pivot ?? null;
  const pivotTarget = live?.pivotTarget ?? null;
  if (live?.quat) {
    _pcSkinQuat.set(live.quat[0], live.quat[1], live.quat[2], live.quat[3]);
  }

  for (let i = 0; i < ids.length; i++) {
    const s = ids[i] * 3;
    _pcSkinVec.set(combined[s] - ijx, combined[s + 1] - ijy, combined[s + 2] - ijz);
    _pcSkinVec.applyMatrix4(_pcSkinMat);
    if (offset) {
      _pcSkinVec.x += offset[0]; _pcSkinVec.y += offset[1]; _pcSkinVec.z += offset[2];
    }
    if (pivot && pivotTarget && live?.quat) {
      _pcSkinVec.x -= pivot[0]; _pcSkinVec.y -= pivot[1]; _pcSkinVec.z -= pivot[2];
      _pcSkinVec.applyQuaternion(_pcSkinQuat);
      _pcSkinVec.x += pivotTarget[0]; _pcSkinVec.y += pivotTarget[1]; _pcSkinVec.z += pivotTarget[2];
    }
    posAttr.setXYZ(i, _pcSkinVec.x, _pcSkinVec.y, _pcSkinVec.z);
  }
  posAttr.needsUpdate = true;
  // Deliberately NOT calling geometry.computeVertexNormals() here: it's an
  // O(triangles) accumulation pass (213k triangles summed across all 7
  // bones, 137k of those on the barely-visible 0.1-opacity thorax alone),
  // and doing that every single frame during a drag was the actual cause of
  // the lag — confirmed by comparing against the reference VTK-based
  // viewer's (ssm_viewer's) update path, which likewise only replaces point
  // positions per interactive update and never recomputes normals live.
  // Shading is very slightly stale while dragging as a result (normals lag
  // one step behind the deformed positions) — self-corrects the instant
  // @change fires the accurate backend call, which fully rebuilds the mesh
  // (see loadBones's full-recreation path) with fresh normals.
}

// Instant, synchronous, client-only mesh + joint-center update — no
// network. Returns false (does nothing) if the one-time client data hasn't
// finished loading yet, so callers can fall back to the backend path.
function applyPcLiveFromSd(sdValues: number[]): boolean {
  if (!pcClientMeta || !pcInfo.value) return false;
  const weights = sdValues.map((sd, i) => sd * (pcInfo.value!.std[i] ?? 0));
  const combined = reconstructPcVertsLive(weights);
  if (!combined) return false;
  const joints = computeLiveJoints(combined);
  if (!joints) return false;
  const bones = pcClientMeta.bones;
  skinBoneLive(bones.thorax, null, combined, thoraxMesh.value);
  skinBoneLive(bones.clav_r, joints.right.clav, combined, clavicleMeshes.right);
  skinBoneLive(bones.clav_l, joints.left.clav, combined, clavicleMeshes.left);
  skinBoneLive(bones.scap_r, joints.right.scap, combined, scapulaMeshes.right);
  skinBoneLive(bones.scap_l, joints.left.scap, combined, scapulaMeshes.left);
  skinBoneLive(bones.hum_r, joints.right.hum, combined, humerusMeshes.right);
  skinBoneLive(bones.hum_l, joints.left.hum, combined, humerusMeshes.left);

  // Origin markers (gold dots + text labels, shown when guides are on) —
  // updateOriginLabels() already runs every frame and reads whatever world
  // position these markers currently hold, so just moving them here is
  // enough to keep guides tracking the live joint positions too. Thorax's
  // origin is always (0,0,0), never needs updating.
  if (originMarkers.clavicle.right) originMarkers.clavicle.right.position.set(...joints.right.scJoint);
  if (originMarkers.clavicle.left)  originMarkers.clavicle.left.position.set(...joints.left.scJoint);
  if (originMarkers.scapula.right) originMarkers.scapula.right.position.set(...joints.right.acJoint);
  if (originMarkers.scapula.left)  originMarkers.scapula.left.position.set(...joints.left.acJoint);
  if (originMarkers.humerus.right) originMarkers.humerus.right.position.set(...joints.right.ghJoint);
  if (originMarkers.humerus.left)  originMarkers.humerus.left.position.set(...joints.left.ghJoint);

  return true;
}

// Sends the current PC weights to the backend for reconstruction (~0.7s
// round trip — the backend replays the new mesh onto a frozen reference
// skeleton rather than re-solving joints, see server.py). Bound to slider
// @input (fires continuously while dragging) via onPcSliderInput's debounce
// below, so this itself just needs to (a) never run two requests at once,
// since predict_pc mutates this session's own out_ply/bones.json files, and
// (b) not silently drop a slider move that arrived while a request was
// in flight — the `pending` loop below re-sends once more with the latest
// position when that happens, instead of queuing one request per move.
let pcUpdatePending = false;
async function updatePcShape() {
  if (isPcUpdating.value) { pcUpdatePending = true; return; }
  isPcUpdating.value = true;
  do {
    pcUpdatePending = false;
    statusMessage.value = "Updating shape...";
    statusColor.value = "#ffffff";
    try {
      const pcWeights = pcSd.value.map((sd, i) => sd * (pcInfo.value?.std[i] ?? 0));
      const response = await fetch("/api/predict_pc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, pc_weights: pcWeights }),
      });
      if (!response.ok) throw await response.text();
      const boneData = await response.json();
      statusMessage.value = "Shape updated.";
      statusColor.value = "#48c774";
      loadBones(boneData, 'pcShape');
    } catch (error) {
      statusMessage.value = "Shape update failed: " + error;
      statusColor.value = "#f14668";
    }
  } while (pcUpdatePending);
  isPcUpdating.value = false;
}

// While dragging: if the client-side preview data (fetchPcClientData) has
// finished loading, just mark the mesh dirty — the animate() loop applies
// it once per rendered frame (see there for why: 'input' events can fire
// much faster than the screen redraws, and applying on every single one
// was still recomputing normals over ~107k vertices far more often than
// necessary, which is what made even the client-side path feel laggy).
// Otherwise fall back to the original debounced-backend path so dragging
// before the one-time fetch completes still does something. Either way,
// @change on the same <input> (see template) fires the accurate backend
// call once on release, correcting the client path's frozen-joint
// approximation and refreshing joint markers/guides.
let pcInputTimer: ReturnType<typeof setTimeout> | undefined;
function onPcSliderInput() {
  if (pcClientMeta && pcInfo.value) {
    pcLiveDirty = true;
    return;
  }
  if (pcInputTimer) clearTimeout(pcInputTimer);
  pcInputTimer = setTimeout(() => { pcInputTimer = undefined; updatePcShape(); }, 120);
}

function resetPcShape() {
  if (!pcInfo.value) return;
  pcSd.value = new Array(pcInfo.value.n_modes).fill(0);
  updatePcShape();
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
  statusColor.value = "#00caef";

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
        fabrik_step: 4,
        // Always a real solve — this dev-only button exists specifically to
        // test FABRIK, regardless of the fast-prediction Settings toggle.
        use_full_solve: true,
      })
    });

    if (!response.ok) throw await response.text();
    const boneData = await response.json();
    predictionProgress.value = 100;
    statusMessage.value = "FABRIK Complete!";
    statusColor.value = "#48c774";
    loadBones(boneData, 'predicted');
  } catch (error) {
    statusMessage.value = "FABRIK Failed: " + error;
    statusColor.value = "#f14668";
  }

  isPredicting.value = false;
}

// Pick a model from the viewport dropdown — mean, the anthropometric
// prediction, or the PC-shape adjustment, whichever are available. Each
// keeps its own data slot (see loadBones), so switching between them never
// discards the others. Re-renders only when the choice actually changes;
// loadBones() with no args re-renders from the slot viewMode now points at
// (or re-fetches /bones.json for 'mean').
function selectModel(mode: ViewMode) {
  isModelListOpen.value = false;
  if (viewMode.value === mode) return;
  viewMode.value = mode;
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

           <!-- Floating toolbar — opens the panel straight to a given view. -->
           <div v-show="!isPanelOpen" class="viewport-tools">
              <!-- Blender-style axis gizmo — click an axis to snap the view. -->
              <canvas ref="gizmoCanvas" class="view-gizmo" title="Click an axis to orient the view"></canvas>
              <button @click="isPanelOpen = true; isKinematicVisible = false; isPcVisible = false; isSettingsVisible = false" class="tool-btn" title="Open Shoulder Predictor" aria-label="Open Shoulder Predictor">
                 <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
              </button>
              <button v-if="SHOW_KINEMATICS_TAB" @click="isPanelOpen = true; isKinematicVisible = true; isPcVisible = false; isSettingsVisible = false" class="tool-btn" title="Kinematics" aria-label="Kinematics">
                 <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 11v8a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h15.5a2.5 2.5 0 0 1 0 5H6"></path><path d="M10 11v8a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-8"></path><path d="M10 11h4"></path></svg>
              </button>
              <button @click="isPanelOpen = true; isPcVisible = true; isKinematicVisible = false; isSettingsVisible = false; fetchPcInfo(); fetchPcClientData()" class="tool-btn" title="Shape (PCA)" aria-label="Shape (PCA)">
                 <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="6" x2="20" y2="6"></line><circle cx="14" cy="6" r="2" fill="currentColor"></circle><line x1="4" y1="12" x2="20" y2="12"></line><circle cx="8" cy="12" r="2" fill="currentColor"></circle><line x1="4" y1="18" x2="20" y2="18"></line><circle cx="16" cy="18" r="2" fill="currentColor"></circle></svg>
              </button>
              <button @click="isPanelOpen = true; isSettingsVisible = true; isKinematicVisible = false; isPcVisible = false" class="tool-btn" title="Application Settings" aria-label="Application Settings">
                 <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
              </button>
           </div>

           <!-- Viewport Overlay Label — doubles as a model picker dropdown -->
           <div class="model-selector animate-in">
              <button class="viewport-label" :class="{ open: isModelListOpen }" @click="isModelListOpen = !isModelListOpen" title="Switch model">
                 <div class="status-indicator" :class="{ active: viewMode !== 'mean' }"></div>
                 <span class="label-text">
                   {{ viewMode === 'mean' ? 'Mean Anatomical Model' : (viewMode === 'predicted' ? 'Predicted Patient-Specific Mesh' : 'Shape-Adjusted Mesh') }}
                 </span>
                 <svg class="chevron" :class="{ flipped: isModelListOpen }" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
              </button>
              <div v-if="isModelListOpen" class="model-list">
                 <button class="model-option" :class="{ active: viewMode === 'mean' }" @click="selectModel('mean')">
                    <div class="status-indicator"></div>
                    <span>Mean Anatomical Model</span>
                 </button>
                 <button v-if="hasPrediction" class="model-option" :class="{ active: viewMode === 'predicted' }" @click="selectModel('predicted')">
                    <div class="status-indicator active"></div>
                    <span>Predicted Patient-Specific Mesh</span>
                 </button>
                 <button v-if="hasPcShape" class="model-option" :class="{ active: viewMode === 'pcShape' }" @click="selectModel('pcShape')">
                    <div class="status-indicator active"></div>
                    <span>Shape-Adjusted Mesh</span>
                 </button>
                 <div v-if="!hasPrediction && !hasPcShape" class="model-empty">Run a prediction or adjust PC sliders to add more models.</div>
              </div>
           </div>
           <!-- Click-away catcher for the model dropdown -->
           <div v-if="isModelListOpen" class="dropdown-backdrop" @pointerdown="isModelListOpen = false"></div>

           <div class="frame-reflection"></div>
        </div>
     </div>

    <!-- Tap anywhere on the viewport (outside the panel) to dismiss the drawer. -->
    <div v-if="isPanelOpen" class="panel-backdrop" @pointerdown="isPanelOpen = false"></div>

    <div class="right-pane" :class="{ open: isPanelOpen }">
      <div class="viewer-wrapper">
         <div class="floating-frame right-content">
            <div class="pane-header">
              <h2>{{ isSettingsVisible ? 'Application Settings' : (isKinematicVisible ? 'Kinematics' : (isPcVisible ? 'Shape (PCA) Adjustment' : 'Shoulder Predictor')) }}</h2>
              <div class="header-actions">
                <div v-if="isPcVisible" class="pc-header-row">
                  <button class="icon-btn pc-info-btn" @click="isPcInfoOpen = !isPcInfoOpen" title="About this tab" aria-label="About this tab">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                  </button>
                  <div v-if="isPcInfoOpen" class="pc-info-popup">
                    <p class="hint">Adjust individual shape-model principal components to see how each one deforms the mesh. This is a debug view — every joint stays fixed exactly as placed on the mean shape; only the bone surfaces themselves follow the sliders.</p>
                  </div>
                </div>
                <button @click="isPanelOpen = false" class="icon-btn" title="Close panel" aria-label="Close panel">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                </button>
              </div>
            </div>
            <div v-if="isPcInfoOpen" class="dropdown-backdrop" @pointerdown="isPcInfoOpen = false"></div>

            <div v-if="isSettingsVisible" class="settings-view animate-in">
              <div class="card transparent-card">
                <h3 style="color: #f59e0b">🎨 Appearance</h3>
                <p class="hint">Choose a colour theme for the interface.</p>
                <div class="toggle-group" style="color: white; display: flex; align-items: center; gap: 10px;">
                  <input type="checkbox" :checked="!isLightMode" @change="toggleTheme" id="darkModeToggle" />
                  <label for="darkModeToggle">Dark Mode</label>
                </div>
              </div>
              <div class="card transparent-card">
                <h3 style="color: #60A060">🔍 Anatomical Guides</h3>
                <p class="hint">Overlay anatomical references on the 3D model.</p>
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
                <div class="toggle-group" style="margin-top: 5px; color: white; display: flex; align-items: center; gap: 10px;">
                  <input type="checkbox" v-model="isMusclePointsEnabled" @change="loadBones()" id="musclePointsToggle" />
                  <label for="musclePointsToggle">Show Muscle Points</label>
                </div>
              </div>
              <div class="card transparent-card">
                <h3 style="color: #00caef">🦴 Prediction Accuracy</h3>
                <p class="hint">Choose how the anthropometric prediction's joint pose is solved.</p>
                <div class="toggle-group" style="color: white; display: flex; align-items: center; gap: 10px;">
                  <input type="checkbox" v-model="useFullJointSolve" id="fullSolveToggle" />
                  <label for="fullSolveToggle">Solve full per-person joint pose (slower)</label>
                </div>
                <p class="hint" style="margin-top: 6px;">Off (default): fast — reuses the mean model's joint orientation, only joint positions are personalized. On: derives this person's own joint orientation too, via a full solve that can take significantly longer.</p>
              </div>
              <div class="card transparent-card">
                <h3 style="color: #3d49d8">📂 Backend Path Configuration</h3>
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
                <h3 style="color: #00caef">🦴 Kinematics</h3>
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

            <div v-else-if="isPcVisible" class="settings-view animate-in kinematic-scroll">
              <div class="card transparent-card">
                <div v-if="isPcLoading" class="hint">Loading shape model (first time can take ~1-2 min)...</div>
                <div v-else-if="!pcInfo" class="hint">Failed to load shape model. Re-open this tab to retry.</div>

                <div v-else class="sub-group">
                  <div v-for="(sd, i) in pcSd" :key="i" class="slider-row">
                    <label><span>PC{{ i + 1 }} ({{ pcInfo.variance_pct[i].toFixed(1) }}% variance)</span> <span>{{ sd.toFixed(2) }} SD</span></label>
                    <input
                      type="range"
                      v-model.number="pcSd[i]"
                      :min="-PC_SD_RANGE"
                      :max="PC_SD_RANGE"
                      :step="0.05"
                      @input="onPcSliderInput"
                      @change="updatePcShape"
                    />
                  </div>
                </div>
              </div>

              <div class="footer-actions">
                <button :disabled="!pcInfo || isPcUpdating" @click="resetPcShape" class="run-btn step-btn">
                  <span v-if="!isPcUpdating">↺ Reset to Mean Shape</span>
                  <span v-else>🔄 Updating...</span>
                </button>
              </div>

              <button @click="isPcVisible = false" class="secondary-btn">Close</button>
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
                </div>

                <!-- Bone measurements: the guide card reorders (via :style order)
                     to sit directly under whichever field is focused, and the
                     field scrolls up so both clear the on-screen keyboard. -->
                <div class="bone-fields">
                  <div class="bf" style="order: 1">
                    <label>R Clavicle Length (mm)</label>
                    <input v-model="r_clav_len" class="input-fi" @focus="onMeasureFocus('clav', $event)" @blur="onMeasureBlur()" />
                  </div>
                  <div class="bf" style="order: 3">
                    <label>R Humerus Length (mm)</label>
                    <input v-model="r_hum_len" class="input-fi" @focus="onMeasureFocus('hum', $event)" @blur="onMeasureBlur()" />
                  </div>
                  <div class="bf" style="order: 5">
                    <label>R Hum Epicondyle Width (mm)</label>
                    <input v-model="r_hum_epi_width" class="input-fi" @focus="onMeasureFocus('epi', $event)" @blur="onMeasureBlur()" />
                  </div>

                  <div v-show="showMeasureGuide" class="measure-guide-card animate-in" :style="{ order: guideOrder }">
                    <h3 style="color: #60A060">📏 How to Measure</h3>
                    <p class="hint">The highlighted bone shows where to take this measurement.</p>
                    <div class="measure-diagram">
                      <div class="body-figure">
                        <img :src="guideImage" @error="onGuideError" :style="guideImgStyle" :alt="measureInfo.name + ' highlighted on the skeleton'" />
                      </div>
                    </div>
                    <div class="measure-active">
                      <span class="measure-name">{{ measureInfo.name }}</span>
                      <span class="measure-desc">{{ measureInfo.desc }}</span>
                    </div>
                  </div>
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
  background: radial-gradient(circle at 50% 50%, #14144f 0%, #0c0c48 100%);
}
</style>

<style scoped>
.container {
  display: flex;
  height: 100vh;
  width: 100vw;
  background: transparent;
  color: #e0e0e0;
  font-family: 'Inter', system-ui, sans-serif;
}
.left-pane {
  flex: 1; /* Full-screen 3D viewport */
  background: transparent;
  overflow: hidden;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}
.viewer-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}
/* Full-bleed viewport: drop the inset/rounding so the model fills the window. */
.left-pane .floating-frame {
  border-radius: 0;
  border: none;
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
  position: fixed;
  top: 0;
  right: 0;
  height: 100vh;
  width: min(460px, 92vw);
  padding: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  background: transparent;
  align-items: stretch;
  justify-content: stretch;
  z-index: 50;
  transform: translateX(100%);
  transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
.right-pane.open {
  transform: translateX(0);
}
/* Invisible scrim over the viewport while the drawer is open; a press dismisses
   it. Sits above the viewport toolbar (z 20) but below the drawer (z 50), so the
   model stays visible and taps on the panel itself don't close it. */
.panel-backdrop {
  position: fixed;
  inset: 0;
  z-index: 40;
  background: transparent;
  cursor: pointer;
}
/* In drawer mode the panel fills its full height, edge to edge. */
.right-pane .viewer-wrapper {
  width: 100%;
  height: 100%;
}
.right-content {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-radius: 0;
  box-shadow: -20px 0 60px -15px rgba(0, 0, 0, 0.7);
}
.pane-header {
  padding: 15px 25px;
  background: rgba(26, 26, 46, 0.4);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px 14px;
  /* In the narrow drawer, let the actions drop to a second row under the title
     rather than clip — but keep them together on one line (see header-actions). */
  flex-wrap: wrap;
}
.pane-header h2 {
  min-width: 0;
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  flex-wrap: nowrap;
  justify-content: flex-end;
  margin-left: auto;
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
  background: #14144f;
  padding: 18px;
  border-radius: 12px; /* Material */
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
.measure-diagram {
  display: flex;
  justify-content: center;
  padding: 8px 0 14px;
}
/* Pre-cropped skeleton illustration (right shoulder + arm to the elbow).
   max-width + an optional inline max-height let it shrink to fit above the
   on-screen keyboard while preserving aspect ratio. */
.body-figure {
  width: 200px;
  text-align: center;
}
.body-figure img {
  max-width: 100%;
  display: inline-block;
  border-radius: 10px;
  vertical-align: top;
}
.measure-active {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-left: 12px;
  border-left: 2px solid rgba(96, 160, 96, 0.5);
}
.measure-name {
  font-size: 0.9rem;
  font-weight: 700;
  color: #7fbf7f;
}
.measure-desc {
  font-size: 0.8rem;
  line-height: 1.45;
  color: #b0b0c0;
}
.light-mode .measure-name { color: #3f8f3f; }
.light-mode .measure-desc { color: #475569; }
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
  border-radius: 12px; /* Material */
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
/* Bone-measurement fields stacked full-width; the guide card reorders between
   them (via inline `order`) to sit right under the focused field. */
.bone-fields {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 12px;
}
.measure-guide-card {
  padding: 14px 16px;
  border: 1px solid rgba(96, 160, 96, 0.4);
  border-radius: 12px;
  background: rgba(96, 160, 96, 0.07);
}
.measure-guide-card h3 {
  margin-bottom: 6px;
}
.light-mode .measure-guide-card {
  background: rgba(96, 160, 96, 0.1);
  border-color: rgba(96, 160, 96, 0.35);
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
.pc-header-row {
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
}
.pc-info-btn {
  padding: 6px;
}
.pc-info-popup {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  z-index: 15;
  width: min(320px, 80vw);
  padding: 12px 14px;
  background: rgba(15, 23, 42, 0.9);
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.12);
  animation: fadeIn 0.15s ease-out;
}
.pc-info-popup .hint {
  margin: 0;
}
.light-mode .pc-info-popup {
  background: rgba(255, 255, 255, 0.95);
  border-color: rgba(0, 0, 0, 0.1);
}
/* Floating toolbar — stacks over the full-screen viewport to open the panel. */
.viewport-tools {
  position: absolute;
  top: 20px;
  right: 20px;
  z-index: 20;
  display: flex;
  flex-direction: column;
  align-items: flex-end; /* keep icon buttons at 46px, right-aligned under the gizmo */
  gap: 12px;
}
.view-gizmo {
  width: 128px;
  height: 128px;
  cursor: pointer;
}
.tool-btn {
  width: 46px;
  height: 46px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(26, 26, 46, 0.55);
  color: #e0e0e0;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 12px;
  cursor: pointer;
  backdrop-filter: blur(8px);
  box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.6);
  transition: all 0.2s;
}
.tool-btn:hover {
  background: rgba(62, 62, 102, 0.7);
  color: #fff;
  transform: translateY(-1px);
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
  border-radius: 12px; /* Material */
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
  border-radius: 12px; /* Material */
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
  accent-color: #00caef;
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
  color: #00caef;
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
  background: linear-gradient(135deg, #3d49d8 0%, #1f2bd4 100%);
  box-shadow: 0 10px 20px rgba(59, 130, 246, 0.2);
}
.save-btn:hover:not(:disabled) {
  box-shadow: 0 12px 24px rgba(59, 130, 246, 0.3);
}

.model-selector {
  position: absolute;
  top: 30px;
  left: 30px;
  z-index: 15;
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: flex-start;
}
.viewport-label {
  padding: 10px 20px;
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  cursor: pointer;
  font-family: inherit;
  transition: background 0.2s, border-color 0.2s;
}
.viewport-label:hover,
.viewport-label.open {
  background: rgba(30, 41, 59, 0.75);
  border-color: rgba(255, 255, 255, 0.2);
}
.chevron {
  color: #94a3b8;
  transition: transform 0.25s ease;
}
.chevron.flipped {
  transform: rotate(180deg);
}
/* Transparent frame listing the available models. */
.model-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 6px;
  min-width: 260px;
  background: rgba(15, 23, 42, 0.35);
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.12);
  animation: fadeIn 0.2s ease-out;
}
.model-option {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: transparent;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.03rem;
  color: #cbd5e1;
  text-align: left;
  transition: background 0.15s, color 0.15s;
}
.model-option:hover {
  background: rgba(255, 255, 255, 0.08);
  color: #f8fafc;
}
.model-option.active {
  background: rgba(31, 43, 212, 0.25);
  color: #f8fafc;
}
.model-empty {
  padding: 8px 14px;
  font-size: 0.75rem;
  color: #94a3b8;
}
.dropdown-backdrop {
  position: fixed;
  inset: 0;
  z-index: 14;
  background: transparent;
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
  background: #1f2bd4;
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
.light-mode .tool-btn {
  background: rgba(255, 255, 255, 0.75);
  color: #475569;
  border: 1px solid rgba(0, 0, 0, 0.1);
}
.light-mode .tool-btn:hover {
  background: #ffffff;
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
.light-mode .viewport-label:hover,
.light-mode .viewport-label.open {
  background: rgba(255, 255, 255, 0.9);
  border-color: rgba(0, 0, 0, 0.15);
}
.light-mode .label-text { color: #0f172a; }
.light-mode .model-list {
  background: rgba(255, 255, 255, 0.55);
  border: 1px solid rgba(255, 255, 255, 0.6);
  box-shadow: 0 8px 32px rgba(30, 41, 59, 0.15), inset 0 1px 0 rgba(255,255,255,0.7);
}
.light-mode .model-option { color: #475569; }
.light-mode .model-option:hover {
  background: rgba(15, 23, 42, 0.06);
  color: #0f172a;
}
.light-mode .model-option.active {
  background: rgba(31, 43, 212, 0.12);
  color: #0f172a;
}
.light-mode .model-empty { color: #64748b; }

/* ── Touch targets (tablet, landscape) ─────────────────────────────────────
   Enlarge the fiddly native controls so they're finger-friendly. */
button, .icon-btn, .secondary-btn, .run-btn {
  min-height: 44px;
  touch-action: manipulation;
}
input[type="number"], input[type="text"], select {
  min-height: 40px;
  font-size: 16px;         /* >=16px stops iOS auto-zooming the field on focus */
  touch-action: manipulation;
}
input[type="checkbox"] {
  width: 24px;
  height: 24px;
  touch-action: manipulation;
}
/* Bigger slider hit area + thumb for dragging with a fingertip */
input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  height: 34px;
  touch-action: pan-y;      /* let vertical page-scroll through, drag horizontally */
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: #00caef;
  border: 2px solid #fff;
  margin-top: -10px;
}
input[type="range"]::-webkit-slider-runnable-track {
  height: 6px;
  border-radius: 3px;
  background: rgba(0, 202, 239, 0.35);
}
input[type="range"]::-moz-range-thumb {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: #00caef;
  border: 2px solid #fff;
}
input[type="range"]::-moz-range-track {
  height: 6px;
  border-radius: 3px;
  background: rgba(0, 202, 239, 0.35);
}
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
