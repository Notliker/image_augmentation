<script setup>
import { computed, onMounted, ref, watch } from 'vue'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const catalog = ref([])
const groupedCatalog = computed(() => {
  return catalog.value.reduce((acc, item) => {
    const bucket = item.category || 'misc'
    if (!acc[bucket]) acc[bucket] = []
    acc[bucket].push(item)
    return acc
  }, {})
})

const selectedAugName = ref('')
const selectedDefinition = computed(() => catalog.value.find((item) => item.name === selectedAugName.value))
const newStepParams = ref({})
const steps = ref([])

const selectedFile = ref(null)
const originalPreview = ref('')
const resultPreview = ref('')
const lastShape = ref(null)

const loadingCatalog = ref(false)
const processing = ref(false)
const status = ref('')
const statusTone = ref('neutral')

const pipelineJson = computed(() => JSON.stringify(steps.value, null, 2))

const setStatus = (message, tone = 'neutral') => {
  status.value = message
  statusTone.value = tone
}

const deepClone = (obj) => JSON.parse(JSON.stringify(obj))

const buildParamDefaults = (definition) => {
  if (!definition?.params) return {}
  const result = {}
  definition.params.forEach((param) => {
    if (param.type === 'array') {
      if (Array.isArray(param.default)) {
        result[param.key] = [...param.default]
      } else if (param.length) {
        result[param.key] = Array(param.length).fill(0)
      } else {
        result[param.key] = []
      }
    } else if (param.type === 'text') {
      if (param.default && typeof param.default === 'object') {
        result[param.key] = JSON.stringify(param.default)
      } else {
        result[param.key] = param.default ?? ''
      }
    } else {
      result[param.key] = param.default ?? null
    }
  })
  return result
}

watch(
  selectedDefinition,
  (def) => {
    newStepParams.value = buildParamDefaults(def)
  },
  { immediate: true },
)

const handleFile = (event) => {
  const file = event.target.files?.[0]
  if (!file) return
  selectedFile.value = file

  const reader = new FileReader()
  reader.onload = (e) => {
    originalPreview.value = e.target?.result
  }
  reader.readAsDataURL(file)
  setStatus('Image loaded', 'info')
}

const resetFile = () => {
  selectedFile.value = null
  originalPreview.value = ''
  resultPreview.value = ''
  lastShape.value = null
}

const addStep = () => {
  if (!selectedAugName.value) {
    setStatus('Pick an augmentation first', 'error')
    return
  }
  const definition = selectedDefinition.value
  const step = { name: selectedAugName.value, params: deepClone(newStepParams.value) }
  steps.value.push(step)
  newStepParams.value = buildParamDefaults(definition)
  setStatus('Step added', 'success')
}

const removeStep = (index) => {
  steps.value.splice(index, 1)
}

const moveStep = (index, direction) => {
  const target = index + direction
  if (target < 0 || target >= steps.value.length) return
  const updated = [...steps.value]
  ;[updated[index], updated[target]] = [updated[target], updated[index]]
  steps.value = updated
}

const normalizeParamsForSend = (stepName, params) => {
  const definition = catalog.value.find((item) => item.name === stepName)
  const normalized = {}

  Object.entries(params || {}).forEach(([key, value]) => {
    const paramMeta = definition?.params?.find((p) => p.key === key)
    if (value === '' || value === undefined || value === null) {
      return
    }

    if (paramMeta?.type === 'array' && Array.isArray(value)) {
      normalized[key] = value.map((v) => (typeof v === 'string' && v !== '' ? Number(v) : v))
      return
    }

    if (paramMeta?.type === 'text' && typeof value === 'string') {
      const trimmed = value.trim()
      if (!trimmed) return
      if (key === 'inner_params') {
        try {
          normalized[key] = JSON.parse(trimmed)
          return
        } catch (e) {
          normalized[key] = trimmed
          return
        }
      }
      normalized[key] = trimmed
      return
    }

    normalized[key] = value
  })

  return normalized
}

const runPipeline = async () => {
  if (!selectedFile.value) {
    setStatus('Please upload an image first', 'error')
    return
  }

  processing.value = true
  setStatus('Sending to backend...', 'info')

  const payloadSteps = steps.value.map((step) => ({
    name: step.name,
    params: normalizeParamsForSend(step.name, step.params),
  }))

  const formData = new FormData()
  formData.append('file', selectedFile.value)
  formData.append('config', JSON.stringify(payloadSteps))

  try {
    const response = await fetch(`${API_BASE}/process`, { method: 'POST', body: formData })
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}))
      const detail = errorBody.detail || response.statusText
      throw new Error(detail)
    }
    const data = await response.json()
    resultPreview.value = `data:${data.mime_type};base64,${data.image_base64}`
    lastShape.value = data.shape
    setStatus('Processing complete', 'success')
  } catch (err) {
    console.error(err)
    setStatus(err.message || 'Processing failed', 'error')
  } finally {
    processing.value = false
  }
}

const fetchCatalog = async () => {
  loadingCatalog.value = true
  try {
    const response = await fetch(`${API_BASE}/augmentations`)
    if (!response.ok) throw new Error('Failed to load augmentations')
    const data = await response.json()
    catalog.value = data.items || []
    if (!selectedAugName.value && catalog.value.length) {
      selectedAugName.value = catalog.value[0].name
    }
    setStatus('Augmentations loaded', 'info')
  } catch (err) {
    console.error(err)
    setStatus('Could not load augmentations', 'error')
  } finally {
    loadingCatalog.value = false
  }
}

onMounted(fetchCatalog)
</script>

<template>
  <div class="page">
    <header class="hero">
      <div>
        <p class="eyebrow">Image playground</p>
        <h1>Augment images with your pipeline</h1>
        <p class="lede">
          Upload a picture, build a chain of augmentations, and preview the result instantly. The list of operations
          comes from the Python pipeline on the server.
        </p>
      </div>
      <div class="hero-actions">
        <button class="primary" type="button" @click="runPipeline" :disabled="processing || !selectedFile">
          {{ processing ? 'Working...' : 'Run pipeline' }}
        </button>
        <p class="status" :data-tone="statusTone" v-if="status">{{ status }}</p>
      </div>
    </header>

    <main class="layout">
      <section class="panel controls">
        <div class="block">
          <div class="block-header">
            <h2>1 · Upload image</h2>
            <button class="ghost" type="button" @click="resetFile" v-if="selectedFile">Reset</button>
          </div>
          <label class="upload">
            <input type="file" accept="image/*" @change="handleFile" />
            <div>
              <p class="upload-title">{{ selectedFile ? selectedFile.name : 'Drop or choose an image' }}</p>
              <p class="upload-hint">Supported formats: jpg, png, bmp</p>
            </div>
          </label>
        </div>

        <div class="block">
          <div class="block-header">
            <h2>2 · Build pipeline</h2>
            <span class="hint">{{ steps.length }} step(s)</span>
          </div>
          <div class="stack">
            <label class="field">
              <span>Augmentation</span>
              <select v-model="selectedAugName" :disabled="loadingCatalog">
                <option value="" disabled>Select...</option>
                <option v-for="item in catalog" :key="item.name" :value="item.name">
                  {{ item.title }} · {{ item.category }}
                </option>
              </select>
            </label>

            <div v-if="selectedDefinition" class="param-grid">
              <div v-for="param in selectedDefinition.params" :key="param.key" class="field">
                <div class="label-row">
                  <span>{{ param.label }}</span>
                  <small v-if="param.hint" class="hint">{{ param.hint }}</small>
                </div>

                <input
                  v-if="param.type === 'number'"
                  type="number"
                  v-model.number="newStepParams[param.key]"
                  :min="param.min"
                  :max="param.max"
                  :step="param.step || 'any'"
                />

                <select v-else-if="param.type === 'select'" v-model="newStepParams[param.key]">
                  <option v-for="option in param.options" :key="option" :value="option">{{ option }}</option>
                </select>

                <div v-else-if="param.type === 'array'" class="array-field">
                  <input
                    v-for="(val, idx) in newStepParams[param.key]"
                    :key="idx"
                    type="number"
                    v-model.number="newStepParams[param.key][idx]"
                    :placeholder="param.placeholder || `Value ${idx + 1}`"
                  />
                </div>

                <input
                  v-else
                  type="text"
                  v-model="newStepParams[param.key]"
                  :placeholder="param.placeholder || ''"
                />
              </div>
            </div>

            <div class="actions">
              <button type="button" class="primary" @click="addStep" :disabled="!selectedAugName">Add step</button>
              <span class="hint">Steps execute top to bottom</span>
            </div>
          </div>

          <div v-if="steps.length" class="stack">
            <div class="step" v-for="(step, index) in steps" :key="`${step.name}-${index}`">
              <div>
                <p class="step-title">{{ index + 1 }} · {{ step.name }}</p>
                <p class="step-subtitle">{{ Object.keys(step.params || {}).length }} param(s)</p>
              </div>
              <div class="step-actions">
                <button class="ghost" type="button" @click="moveStep(index, -1)">↑</button>
                <button class="ghost" type="button" @click="moveStep(index, 1)">↓</button>
                <button class="ghost danger" type="button" @click="removeStep(index)">✕</button>
              </div>
            </div>

            <div class="code">
              <div class="code-header">
                <span>Payload preview</span>
                <small class="hint">Sent as JSON to /process</small>
              </div>
              <pre>{{ pipelineJson }}</pre>
            </div>
          </div>
        </div>
      </section>

      <section class="panel previews">
        <div class="preview-grid">
          <div class="preview-card">
            <p class="preview-title">Original</p>
            <div class="preview-frame" :data-empty="!originalPreview">
              <img v-if="originalPreview" :src="originalPreview" alt="Original preview" />
              <p v-else class="placeholder">Upload an image to view</p>
            </div>
          </div>

          <div class="preview-card">
            <p class="preview-title">Result</p>
            <div class="preview-frame" :data-empty="!resultPreview">
              <img v-if="resultPreview" :src="resultPreview" alt="Result preview" />
              <p v-else class="placeholder">Run the pipeline to see output</p>
            </div>
            <p v-if="lastShape" class="hint">
              {{ lastShape.width }} × {{ lastShape.height }} · {{ lastShape.channels }} channel(s)
            </p>
          </div>
        </div>

        <div class="block available">
          <div class="block-header">
            <h2>Available augmentations</h2>
            <span class="hint">{{ catalog.length }} items</span>
          </div>
          <div class="pill-row" v-if="Object.keys(groupedCatalog).length">
            <div v-for="(items, category) in groupedCatalog" :key="category" class="pill-group">
              <p class="pill-title">{{ category }}</p>
              <div class="pill-list">
                <span class="pill" v-for="item in items" :key="item.name">{{ item.title }}</span>
              </div>
            </div>
          </div>
          <p v-else class="placeholder">Catalog will appear after backend responds.</p>
        </div>
      </section>
    </main>
  </div>
</template>
