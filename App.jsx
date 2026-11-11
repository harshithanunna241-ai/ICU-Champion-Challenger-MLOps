import { useMemo, useState } from 'react'
import './App.css'

const API_BASE_URL = 'http://127.0.0.1:8002'

const OBSERVATION_VARIABLES = [
  { value: 'heart_rate', label: 'Heart Rate (bpm)' },
  { value: 'systolic_bp', label: 'Systolic BP (mmHg)' },
  { value: 'diastolic_bp', label: 'Diastolic BP (mmHg)' },
  { value: 'mean_bp', label: 'Mean BP (mmHg)' },
  { value: 'respiratory_rate', label: 'Respiratory Rate (/min)' },
  { value: 'temperature', label: 'Temperature (°C)' },
  { value: 'spo2', label: 'SpO₂ (%)' },
  { value: 'glucose', label: 'Glucose' },
]

function getLocalDateTimeValue(date = new Date()) {
  const localDate = new Date(
    date.getTime() - date.getTimezoneOffset() * 60000
  )

  return localDate.toISOString().slice(0, 16)
}

function createObservation() {
  return {
    id: `${Date.now()}-${Math.random()}`,
    charttime: getLocalDateTimeValue(
      new Date(Date.now() - 60 * 60 * 1000)
    ),
    variable: 'heart_rate',
    value: '',
  }
}

const navigation = [
  { label: 'Dashboard', icon: '▦' },
  { label: 'Patient Monitoring', icon: '♡' },
  { label: 'Deterioration Prediction', icon: '⌁' },
  { label: 'Model Leaderboard', icon: '▤' },
  { label: 'Champion–Challenger', icon: '⇄' },
  { label: 'Model Monitoring', icon: '◷' },
  { label: 'Audit Logs', icon: '☷' },
]

const patients = [
  { id: 'ICU-001', name: 'Patient 001', age: 65, heartRate: 112, oxygen: 91, bp: '92/60', risk: 'High', temperature: 38.5, respiratoryRate: 26 },
  { id: 'ICU-002', name: 'Patient 002', age: 54, heartRate: 88, oxygen: 97, bp: '120/80', risk: 'Low', temperature: 36.8, respiratoryRate: 17 },
  { id: 'ICU-003', name: 'Patient 003', age: 71, heartRate: 105, oxygen: 93, bp: '100/68', risk: 'Medium', temperature: 37.8, respiratoryRate: 22 },
  { id: 'ICU-004', name: 'Patient 004', age: 62, heartRate: 76, oxygen: 98, bp: '118/76', risk: 'Low', temperature: 36.6, respiratoryRate: 16 },
]

const models = [
  { name: 'Logistic Regression', type: 'Baseline model', stage: 'Challenger', auroc: '—', auprc: '—', f1: '—', accuracy: '—' },
  { name: 'Random Forest', type: 'Ensemble model', stage: 'Champion', auroc: '—', auprc: '—', f1: '—', accuracy: '—' },
  { name: 'XGBoost', type: 'Boosting model', stage: 'Challenger', auroc: '—', auprc: '—', f1: '—', accuracy: '—' },
]

function App() {
  const [activePage, setActivePage] = useState('Dashboard')

  function navigate(page) {
    setActivePage(page)
  }

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">+</div>
          <div>
            <h2>ICU Intelligence</h2>
            <span>Clinical Decision Platform</span>
          </div>
        </div>

        <div className="nav-heading">WORKSPACE</div>

        <nav className="navigation">
          {navigation.map((item) => (
            <button
              key={item.label}
              className={`nav-item ${activePage === item.label ? 'active' : ''}`}
              onClick={() => navigate(item.label)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot"></span>
            <div>
              <strong>Platform Status</strong>
              <small>UI demonstration mode</small>
            </div>
          </div>
          <div className="profile">
            <div className="avatar">A</div>
            <div>
              <strong>Administrator</strong>
              <small>System User</small>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace / {activePage}</span>
            <h1>{activePage}</h1>
          </div>
          <div className="topbar-right">
            <span className="demo-label">DEMO DATA</span>
            <div className="topbar-date">ICU Monitoring Platform</div>
          </div>
        </header>

        {activePage === 'Dashboard' && <Dashboard onNavigate={navigate} />}
        {activePage === 'Patient Monitoring' && <PatientMonitoring />}
        {activePage === 'Deterioration Prediction' && <DeteriorationPrediction />}
        {activePage === 'Model Leaderboard' && <ModelLeaderboard />}
        {activePage === 'Champion–Challenger' && <ChampionChallenger />}
        {activePage === 'Model Monitoring' && <ModelMonitoring />}
        {activePage === 'Audit Logs' && <AuditLogs />}

        <footer className="footer">
          <span>ICU Patient Deterioration Prediction Platform</span>
          <span>Development Preview · Not for clinical use</span>
        </footer>
      </main>
    </div>
  )
}

/* DASHBOARD */

function Dashboard({ onNavigate }) {
  return (
    <>
      <section className="welcome-section">
        <div>
          <h2>ICU Overview</h2>
          <p>Monitor patient status, deterioration risk, and model performance.</p>
        </div>
        <span className="last-updated">Sample dashboard data</span>
      </section>

      <section className="stats-grid">
        <StatCard label="Total Patients" value="24" detail="Sample ICU census" icon="♙" />
        <StatCard label="High-Risk Patients" value="05" detail="Illustrative category" icon="⚠" tone="red" />
        <StatCard label="Medium-Risk Patients" value="08" detail="Illustrative category" icon="◉" tone="orange" />
        <StatCard label="Active ML Models" value="03" detail="Champion + Challengers" icon="▥" tone="blue" />
      </section>

      <section className="content-grid">
        <div className="panel patients-panel">
          <div className="panel-header">
            <div>
              <h3>Patient Monitoring</h3>
              <p>Illustrative patient vital signs</p>
            </div>
            <button className="text-button" onClick={() => onNavigate('Patient Monitoring')}>View all →</button>
          </div>
          <PatientTable records={patients.slice(0, 4)} />
          <p className="table-note">Fictional demonstration records only.</p>
        </div>

        <div className="panel risk-panel">
          <div className="panel-header">
            <div>
              <h3>Risk Distribution</h3>
              <p>Illustrative categories</p>
            </div>
          </div>
          <div className="risk-chart">
            <div className="donut">
              <div className="donut-center">
                <strong>24</strong>
                <span>Patients</span>
              </div>
            </div>
            <div className="chart-legend">
              <Legend color="red" label="High Risk" value="5" />
              <Legend color="orange" label="Medium Risk" value="8" />
              <Legend color="green" label="Low Risk" value="11" />
            </div>
          </div>
          <p className="table-note">Not live predictions.</p>
        </div>
      </section>

      <section className="panel models-panel">
        <div className="panel-header">
          <div>
            <h3>Model Lifecycle Overview</h3>
            <p>Champion–Challenger model registry</p>
          </div>
          <button className="text-button" onClick={() => onNavigate('Model Leaderboard')}>Model leaderboard →</button>
        </div>
        <div className="models-grid">
          {models.map((model) => (
            <ModelCard key={model.name} model={model} />
          ))}
        </div>
        <p className="table-note">Model metrics are awaiting backend integration.</p>
      </section>
    </>
  )
}

/* PATIENT MONITORING */

function PatientMonitoring() {
  const [search, setSearch] = useState('')
  const [risk, setRisk] = useState('All')
  const [selected, setSelected] = useState(null)

  const filtered = useMemo(() => patients.filter((patient) => {
    const matchesSearch =
      patient.id.toLowerCase().includes(search.toLowerCase()) ||
      patient.name.toLowerCase().includes(search.toLowerCase())

    return matchesSearch && (risk === 'All' || patient.risk === risk)
  }), [search, risk])

  return (
    <div className="page-stack">
      <PageIntro title="Patient Monitoring" description="Search and inspect illustrative ICU patient records and vital signs." />

      <section className="stats-grid">
        <StatCard label="Displayed Patients" value={String(filtered.length).padStart(2, '0')} detail="Current filter results" icon="♙" />
        <StatCard label="High Risk" value="01" detail="Illustrative category" icon="⚠" tone="red" />
        <StatCard label="Medium Risk" value="01" detail="Illustrative category" icon="◉" tone="orange" />
        <StatCard label="Low Risk" value="02" detail="Illustrative category" icon="✓" tone="blue" />
      </section>

      <section className="panel page-panel">
        <div className="panel-header">
          <div>
            <h3>Patient Records</h3>
            <p>Fictional patient information for interface demonstration</p>
          </div>
          <span className="demo-label">DEMO DATA</span>
        </div>

        <div className="toolbar">
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search patient ID or name..."
          />
          <select value={risk} onChange={(event) => setRisk(event.target.value)}>
            <option value="All">All risk levels</option>
            <option value="High">High risk</option>
            <option value="Medium">Medium risk</option>
            <option value="Low">Low risk</option>
          </select>
        </div>

        <PatientTable records={filtered} onSelect={setSelected} showDetails />

        <p className="table-note">Showing {filtered.length} of {patients.length} fictional patient records.</p>
      </section>

      {selected && (
        <section className="panel page-panel">
          <div className="panel-header">
            <div>
              <h3>Patient Details · {selected.id}</h3>
              <p>{selected.name}</p>
            </div>
            <button className="text-button" onClick={() => setSelected(null)}>Close ✕</button>
          </div>
          <div className="detail-grid">
            <Detail label="Age" value={`${selected.age} years`} />
            <Detail label="Heart Rate" value={`${selected.heartRate} bpm`} />
            <Detail label="SpO₂" value={`${selected.oxygen}%`} />
            <Detail label="Blood Pressure" value={selected.bp} />
            <Detail label="Temperature" value={`${selected.temperature} °C`} />
            <Detail label="Respiratory Rate" value={`${selected.respiratoryRate} /min`} />
            <Detail label="Illustrative Risk Category" value={selected.risk} />
          </div>
        </section>
      )}

      <Disclaimer />
    </div>
  )
}

function PatientTable({ records, onSelect, showDetails = false }) {
  return (
    <div className="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Patient ID</th>
            <th>Age</th>
            <th>Heart Rate</th>
            <th>SpO₂</th>
            <th>Blood Pressure</th>
            <th>Risk Level</th>
            {showDetails && <th>Details</th>}
          </tr>
        </thead>
        <tbody>
          {records.map((patient) => (
            <tr key={patient.id}>
              <td className="patient-id">{patient.id}</td>
              <td>{patient.age}</td>
              <td>{patient.heartRate} bpm</td>
              <td>{patient.oxygen}%</td>
              <td>{patient.bp}</td>
              <td><RiskBadge risk={patient.risk} /></td>
              {showDetails && (
                <td>
                  <button className="text-button" onClick={() => onSelect(patient)}>View details</button>
                </td>
              )}
            </tr>
          ))}
          {records.length === 0 && (
            <tr>
              <td colSpan={showDetails ? 7 : 6} className="empty-state">
                No matching patient records.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

/* DETERIORATION PREDICTION - FASTAPI INTEGRATION */

function DeteriorationPrediction() {
  const [form, setForm] = useState(() => ({
    subjectId: '',
    stayId: '',
    predictionTime: getLocalDateTimeValue(),
  }))

  const [observations, setObservations] = useState(() => [
    createObservation(),
  ])

  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const [messageType, setMessageType] = useState('')
  const [serviceStatus, setServiceStatus] = useState('Not checked')
  const [result, setResult] = useState(null)

  function updateForm(event) {
    const { name, value } = event.target

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }))

    setMessage('')
    setResult(null)
  }

  function updateObservation(id, field, value) {
    setObservations((previous) =>
      previous.map((observation) =>
        observation.id === id
          ? { ...observation, [field]: value }
          : observation
      )
    )

    setMessage('')
    setResult(null)
  }

  function addObservation() {
    setObservations((previous) => [
      ...previous,
      createObservation(),
    ])

    setMessage('')
  }

  function removeObservation(id) {
    setObservations((previous) =>
      previous.filter((observation) => observation.id !== id)
    )

    setMessage('')
    setResult(null)
  }

  function clearForm() {
    setForm({
      subjectId: '',
      stayId: '',
      predictionTime: getLocalDateTimeValue(),
    })

    setObservations([createObservation()])
    setMessage('')
    setMessageType('')
    setResult(null)
  }

  async function submit(event) {
    event.preventDefault()

    setMessage('')
    setMessageType('')
    setResult(null)

    if (!Number.isInteger(Number(form.subjectId)) || Number(form.subjectId) <= 0) {
      setMessage('Enter a valid positive patient ID.')
      setMessageType('error')
      return
    }

    if (!Number.isInteger(Number(form.stayId)) || Number(form.stayId) <= 0) {
      setMessage('Enter a valid positive ICU stay ID.')
      setMessageType('error')
      return
    }

    if (!form.predictionTime) {
      setMessage('Select a prediction timestamp.')
      setMessageType('error')
      return
    }

    if (observations.length === 0) {
      setMessage('Add at least one clinical observation.')
      setMessageType('error')
      return
    }

    const predictionDate = new Date(form.predictionTime)

    if (Number.isNaN(predictionDate.getTime())) {
      setMessage('The prediction timestamp is invalid.')
      setMessageType('error')
      return
    }

    const preparedObservations = []

    for (let index = 0; index < observations.length; index += 1) {
      const observation = observations[index]
      const chartDate = new Date(observation.charttime)
      const numericValue = Number(observation.value)

      if (!observation.charttime || Number.isNaN(chartDate.getTime())) {
        setMessage(`Enter a valid timestamp for observation ${index + 1}.`)
        setMessageType('error')
        return
      }

      if (chartDate >= predictionDate) {
        setMessage(`Observation ${index + 1} must be earlier than the prediction timestamp.`)
        setMessageType('error')
        return
      }

      const lookbackStart = new Date(
        predictionDate.getTime() - 6 * 60 * 60 * 1000
      )

      if (chartDate < lookbackStart) {
        setMessage(`Observation ${index + 1} is older than the six-hour lookback window.`)
        setMessageType('error')
        return
      }

      if (
        observation.value === '' ||
        !Number.isFinite(numericValue)
      ) {
        setMessage(`Enter a valid numeric value for observation ${index + 1}.`)
        setMessageType('error')
        return
      }

      preparedObservations.push({
        charttime: observation.charttime,
        variable: observation.variable,
        value: numericValue,
      })
    }

    const payload = {
      subject_id: Number(form.subjectId),
      stay_id: Number(form.stayId),
      prediction_time: form.predictionTime,
      observations: preparedObservations,
    }

    setLoading(true)
    setServiceStatus('Checking')

    try {
      const response = await fetch(
        `${API_BASE_URL}/predict/from-observations`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        }
      )

      setServiceStatus('Connected')

      const responseData = await response.json().catch(() => ({}))

      if (!response.ok) {
        const detail =
          typeof responseData.detail === 'string'
            ? responseData.detail
            : 'The prediction request could not be completed.'

        if (response.status === 503) {
          setMessage(
            `FastAPI is reachable, but prediction is currently unavailable: ${detail}`
          )
        } else if (response.status === 422) {
          setMessage(`Request validation failed: ${detail}`)
        } else {
          setMessage(`API error (${response.status}): ${detail}`)
        }

        setMessageType('error')
        return
      }

      setResult(responseData)
      setMessage('Prediction response received successfully.')
      setMessageType('success')
    } catch (error) {
      setServiceStatus('Unavailable')

      if (error instanceof TypeError) {
        setMessage(
          'Cannot connect to FastAPI. Check that the backend is running at http://127.0.0.1:8002.'
        )
      } else {
        setMessage(`Request failed: ${error.message}`)
      }

      setMessageType('error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-stack">
      <PageIntro
        title="Patient Deterioration Assessment"
        description="Submit timestamped ICU observations to the FastAPI prediction service."
      />

      <section className="panel page-panel">
        <div className="panel-header">
          <div>
            <h3>Patient and Prediction Details</h3>
            <p>Enter the patient identifiers and prediction timestamp.</p>
          </div>
          <span className="demo-label">API INTEGRATION</span>
        </div>

        <form onSubmit={submit}>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="subjectId">Patient ID</label>
              <input
                id="subjectId"
                name="subjectId"
                type="number"
                min="1"
                step="1"
                value={form.subjectId}
                onChange={updateForm}
                placeholder="Enter patient ID"
                required
              />
            </div>

            <div className="form-field">
              <label htmlFor="stayId">ICU Stay ID</label>
              <input
                id="stayId"
                name="stayId"
                type="number"
                min="1"
                step="1"
                value={form.stayId}
                onChange={updateForm}
                placeholder="Enter ICU stay ID"
                required
              />
            </div>

            <div className="form-field">
              <label htmlFor="predictionTime">Prediction Timestamp</label>
              <input
                id="predictionTime"
                name="predictionTime"
                type="datetime-local"
                value={form.predictionTime}
                onChange={updateForm}
                required
              />
            </div>
          </div>

          <div className="panel-header" style={{ marginTop: '24px' }}>
            <div>
              <h3>Clinical Observations</h3>
              <p>
                Add measurements recorded during the six hours before prediction.
              </p>
            </div>
            <button
              type="button"
              className="secondary-button"
              onClick={addObservation}
            >
              + Add Observation
            </button>
          </div>

          {observations.map((observation, index) => (
            <div
              key={observation.id}
              className="panel"
              style={{ marginBottom: '14px', padding: '16px' }}
            >
              <div className="panel-header">
                <div>
                  <h3>Observation {index + 1}</h3>
                </div>

                <button
                  type="button"
                  className="text-button"
                  onClick={() => removeObservation(observation.id)}
                  disabled={observations.length === 1}
                  aria-label={`Remove observation ${index + 1}`}
                >
                  Remove
                </button>
              </div>

              <div className="form-grid">
                <div className="form-field">
                  <label htmlFor={`variable-${observation.id}`}>
                    Clinical Variable
                  </label>
                  <select
                    id={`variable-${observation.id}`}
                    value={observation.variable}
                    onChange={(event) =>
                      updateObservation(
                        observation.id,
                        'variable',
                        event.target.value
                      )
                    }
                    required
                  >
                    {OBSERVATION_VARIABLES.map((variable) => (
                      <option key={variable.value} value={variable.value}>
                        {variable.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-field">
                  <label htmlFor={`value-${observation.id}`}>
                    Measurement Value
                  </label>
                  <input
                    id={`value-${observation.id}`}
                    type="number"
                    step="any"
                    value={observation.value}
                    onChange={(event) =>
                      updateObservation(
                        observation.id,
                        'value',
                        event.target.value
                      )
                    }
                    placeholder="Enter measurement"
                    required
                  />
                </div>

                <div className="form-field">
                  <label htmlFor={`time-${observation.id}`}>
                    Measurement Timestamp
                  </label>
                  <input
                    id={`time-${observation.id}`}
                    type="datetime-local"
                    value={observation.charttime}
                    onChange={(event) =>
                      updateObservation(
                        observation.id,
                        'charttime',
                        event.target.value
                      )
                    }
                    required
                  />
                </div>
              </div>
            </div>
          ))}

          <div className="notice">
            The backend uses observations from the same patient and ICU stay
            within the six-hour period before the prediction timestamp.
            Measurements outside this period are not used.
          </div>

          <div className="action-row">
            <button
              type="submit"
              className="primary-button"
              disabled={loading}
            >
              {loading ? 'Generating Prediction...' : 'Generate Prediction'}
            </button>

            <button
              type="button"
              className="secondary-button"
              onClick={clearForm}
              disabled={loading}
            >
              Clear Form
            </button>
          </div>
        </form>

        {message && (
          <div className="notice" role="status" aria-live="polite">
            {message}
          </div>
        )}
      </section>

      <section className="panel page-panel">
        <h3>Prediction Service</h3>

        <p className="service-status">
          <span
            className={`status-dot ${
              serviceStatus === 'Connected'
                ? ''
                : 'pending'
            }`}
          ></span>
          FastAPI connection: {serviceStatus}
        </p>

        <p className="muted">
          API endpoint: POST /predict/from-observations
        </p>

        <p className="muted">
          Backend address: {API_BASE_URL}
        </p>
      </section>

      {result && (
        <section className="panel page-panel">
          <div className="panel-header">
            <div>
              <h3>Prediction Result</h3>
              <p>Response returned by the registered Champion model.</p>
            </div>
            <span className="demo-label">API RESPONSE</span>
          </div>

          <div className="detail-grid">
            <Detail
              label="Patient ID"
              value={result.subject_id ?? form.subjectId}
            />
            <Detail
              label="ICU Stay ID"
              value={result.stay_id ?? form.stayId}
            />
            <Detail
              label="Prediction"
              value={
                result.prediction === 1
                  ? 'Positive'
                  : result.prediction === 0
                    ? 'Negative'
                    : String(result.prediction ?? '—')
              }
            />
            <Detail
              label="Predicted Probability"
              value={
                Number.isFinite(Number(result.probability))
                  ? `${(Number(result.probability) * 100).toFixed(2)}%`
                  : '—'
              }
            />
            <Detail
              label="Model"
              value={result.model_name ?? '—'}
            />
            <Detail
              label="Status"
              value={result.status ?? 'Response received'}
            />
          </div>

          <p className="muted">
            This is an experimental model output, not a clinical diagnosis
            or recommendation.
          </p>
        </section>
      )}

      <Disclaimer />
    </div>
  )
}

/* MODEL LEADERBOARD */

function ModelLeaderboard() {
  const [metric, setMetric] = useState('AUROC')

  return (
    <div className="page-stack">
      <PageIntro title="Model Leaderboard" description="Compare registered model versions using evaluation metrics." />

      <section className="stats-grid">
        <StatCard label="Registered Models" value="03" detail="Illustrative registry" icon="▥" />
        <StatCard label="Champion Models" value="01" detail="Registry stage example" icon="🏆" tone="blue" />
        <StatCard label="Challengers" value="02" detail="Awaiting comparison" icon="⇄" tone="orange" />
        <StatCard label="Best Metric" value="—" detail="Backend data unavailable" icon="◎" tone="green" />
      </section>

      <section className="panel page-panel">
        <div className="panel-header">
          <div>
            <h3>Model Performance Comparison</h3>
            <p>Evaluation results will load from the model registry API.</p>
          </div>
          <span className="demo-label">NOT CONNECTED</span>
        </div>

        <div className="toolbar">
          <label className="toolbar-label" htmlFor="leaderboard-metric">Ranking metric</label>
          <select id="leaderboard-metric" value={metric} onChange={(event) => setMetric(event.target.value)}>
            <option>AUROC</option>
            <option>AUPRC</option>
            <option>F1 Score</option>
            <option>Accuracy</option>
          </select>
        </div>

        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Registry Stage</th>
                <th>AUROC</th>
                <th>AUPRC</th>
                <th>F1 Score</th>
                <th>Accuracy</th>
                <th>Rank</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr key={model.name}>
                  <td className="patient-id">{model.name}</td>
                  <td>{model.type}</td>
                  <td><StageBadge stage={model.stage} /></td>
                  <td>{model.auroc}</td>
                  <td>{model.auprc}</td>
                  <td>{model.f1}</td>
                  <td>{model.accuracy}</td>
                  <td>—</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="table-note">Selected ranking metric: {metric}. Ranking is unavailable until real evaluation results are received.</p>
      </section>

      <section className="panel page-panel">
        <h3>Evaluation Metrics</h3>
        <div className="detail-grid">
          <Detail label="AUROC" value="—" />
          <Detail label="AUPRC" value="—" />
          <Detail label="F1 Score" value="—" />
          <Detail label="Accuracy" value="—" />
          <Detail label="Precision" value="—" />
          <Detail label="Recall" value="—" />
        </div>
        <p className="muted">Metrics are intentionally blank until populated by the backend.</p>
      </section>
    </div>
  )
}

/* CHAMPION–CHALLENGER */

function ChampionChallenger() {
  return (
    <div className="page-stack">
      <PageIntro title="Champion–Challenger Management" description="Review model stages and prepare controlled model promotion workflows." />

      <section className="panel page-panel">
        <div className="panel-header">
          <div>
            <h3>Current Model Registry</h3>
            <p>Illustrative model lifecycle stages</p>
          </div>
          <span className="demo-label">DEMO REGISTRY</span>
        </div>

        <div className="models-grid">
          {models.map((model) => <ModelCard key={model.name} model={model} />)}
        </div>
      </section>

      <section className="panel page-panel">
        <h3>Champion–Challenger Workflow</h3>
        <div className="workflow-grid">
          <WorkflowStep number="01" title="Register Candidate" description="Register a newly trained model with its metadata and evaluation results." />
          <WorkflowStep number="02" title="Evaluate Models" description="Compare candidate and champion using the approved evaluation metrics." />
          <WorkflowStep number="03" title="Review Promotion" description="Check promotion policy, confidence intervals, and validation results." />
          <WorkflowStep number="04" title="Promote or Archive" description="Record the approved lifecycle transition in the model registry." />
        </div>
        <div className="notice">
          Model promotion controls are not enabled in this frontend demonstration. No model lifecycle changes have been made.
        </div>
      </section>

      <section className="panel page-panel">
        <h3>Promotion Readiness</h3>
        <div className="detail-grid">
          <Detail label="Champion model" value="Registry API pending" />
          <Detail label="Challenger evaluation" value="Not loaded" />
          <Detail label="Promotion policy" value="Not evaluated" />
          <Detail label="Approval status" value="Not requested" />
        </div>
      </section>
    </div>
  )
}

/* MODEL MONITORING */

function ModelMonitoring() {
  return (
    <div className="page-stack">
      <PageIntro title="Model Monitoring" description="Monitor model performance, data quality, drift, and operational alerts." />

      <section className="stats-grid">
        <StatCard label="Model Status" value="—" detail="Awaiting monitoring API" icon="◷" />
        <StatCard label="Data Drift" value="—" detail="No live measurement" icon="↗" tone="orange" />
        <StatCard label="Performance Drift" value="—" detail="No live measurement" icon="◎" tone="red" />
        <StatCard label="Active Alerts" value="—" detail="No live alert data" icon="⚠" tone="blue" />
      </section>

      <section className="panel page-panel">
        <div className="panel-header">
          <div>
            <h3>Monitoring Overview</h3>
            <p>Model health and monitoring measurements</p>
          </div>
          <span className="demo-label">NOT CONNECTED</span>
        </div>
        <div className="detail-grid">
          <Detail label="Prediction Volume" value="—" />
          <Detail label="Feature Drift" value="—" />
          <Detail label="Prediction Drift" value="—" />
          <Detail label="Model Performance" value="—" />
          <Detail label="Last Monitoring Run" value="—" />
          <Detail label="Monitoring Window" value="—" />
        </div>
        <p className="muted">Monitoring values will be loaded from the backend. No live health status is currently available.</p>
      </section>

      <section className="panel page-panel">
        <h3>Monitoring Alerts</h3>
        <div className="empty-state-block">
          <div className="placeholder-icon">◷</div>
          <strong>No monitoring alerts loaded</strong>
          <p>Connect the monitoring alerts API to display verified drift and performance events.</p>
        </div>
      </section>

      <section className="panel page-panel">
        <h3>Monitoring Components</h3>
        <div className="workflow-grid">
          <WorkflowStep number="01" title="Data Drift" description="Track changes in incoming feature distributions." />
          <WorkflowStep number="02" title="Prediction Drift" description="Observe changes in model prediction distributions." />
          <WorkflowStep number="03" title="Performance" description="Evaluate performance when verified outcome labels become available." />
          <WorkflowStep number="04" title="Alerting" description="Display threshold-based monitoring events and their status." />
        </div>
      </section>
    </div>
  )
}

/* AUDIT LOGS */

function AuditLogs() {
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState('All')

  const categories = ['All', 'Prediction', 'Model', 'Monitoring', 'User']

  return (
    <div className="page-stack">
      <PageIntro title="Audit Logs" description="Review traceable records of platform activity and model lifecycle events." />

      <section className="stats-grid">
        <StatCard label="Total Events" value="—" detail="Awaiting audit API" icon="☷" />
        <StatCard label="Prediction Events" value="—" detail="Not loaded" icon="⌁" tone="blue" />
        <StatCard label="Model Events" value="—" detail="Not loaded" icon="▥" tone="orange" />
        <StatCard label="User Events" value="—" detail="Not loaded" icon="♙" tone="green" />
      </section>

      <section className="panel page-panel">
        <div className="panel-header">
          <div>
            <h3>Activity History</h3>
            <p>Search and filter audit events returned by the backend.</p>
          </div>
          <span className="demo-label">NOT CONNECTED</span>
        </div>

        <div className="toolbar">
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search audit events..."
          />
          <select value={category} onChange={(event) => setCategory(event.target.value)}>
            {categories.map((item) => <option key={item}>{item}</option>)}
          </select>
        </div>

        <div className="empty-state-block">
          <div className="placeholder-icon">☷</div>
          <strong>No audit records loaded</strong>
          <p>
            {search || category !== 'All'
              ? 'Filters are ready. Matching records will appear when the audit API is connected.'
              : 'Connect the audit logs endpoint to display verified platform activity.'}
          </p>
        </div>
      </section>

      <section className="panel page-panel">
        <h3>Audit Record Fields</h3>
        <div className="detail-grid">
          <Detail label="Event ID" value="—" />
          <Detail label="Timestamp" value="—" />
          <Detail label="Actor" value="—" />
          <Detail label="Action" value="—" />
          <Detail label="Resource" value="—" />
          <Detail label="Event Status" value="—" />
        </div>
        <p className="muted">No audit event values are fabricated for this demonstration.</p>
      </section>
    </div>
  )
}

/* SHARED COMPONENTS */

function PageIntro({ title, description }) {
  return (
    <section className="welcome-section">
      <div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      <span className="last-updated">Development preview</span>
    </section>
  )
}

function StatCard({ label, value, detail, icon, tone = 'purple' }) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${tone}`}>{icon}</div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-detail">{detail}</div>
    </div>
  )
}

function RiskBadge({ risk }) {
  return <span className={`risk-badge ${risk.toLowerCase()}`}>{risk}</span>
}

function StageBadge({ stage }) {
  return <span className={`model-status ${stage.toLowerCase()}`}>{stage}</span>
}

function ModelCard({ model }) {
  return (
    <div className="model-card">
      <div className="model-card-top">
        <span className="model-symbol">ML</span>
        <StageBadge stage={model.stage} />
      </div>
      <h4>{model.name}</h4>
      <p>{model.type}</p>
      <div className="model-metric">
        <span>AUROC</span>
        <strong>{model.auroc}</strong>
      </div>
    </div>
  )
}

function Legend({ color, label, value }) {
  return (
    <div className="legend-item">
      <span className={`legend-dot ${color}`}></span>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function Detail({ label, value }) {
  return (
    <div className="detail-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function Field({ label, name, value, onChange, min, max, step }) {
  return (
    <div className="form-field">
      <label htmlFor={name}>{label}</label>
      <input
        id={name}
        name={name}
        type="number"
        min={min}
        max={max}
        step={step || '1'}
        value={value}
        onChange={onChange}
        placeholder={`Enter ${label.toLowerCase()}`}
        required
      />
    </div>
  )
}

function WorkflowStep({ number, title, description }) {
  return (
    <div className="workflow-step">
      <span>{number}</span>
      <div>
        <strong>{title}</strong>
        <p>{description}</p>
      </div>
    </div>
  )
}

function Disclaimer() {
  return (
    <p className="demo-disclaimer">
      All patient information and vital signs shown in this interface are fictional demonstration data.
      This platform preview is not a medical device and must not be used for clinical decisions.
    </p>
  )
}

export default App