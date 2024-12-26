// App.jsx
import { useState, useRef, useEffect } from 'react'
import axios from 'axios'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [cards, setCards] = useState([])
  const [sets, setSets] = useState([])
  const [scale, setScale] = useState(1)
  const imageRef = useRef(null)

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0])
    if (e.target.files[0]) {
      setImageUrl(URL.createObjectURL(e.target.files[0]))
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const res = await axios.post('http://localhost:5001/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        withCredentials: true,
        crossDomain: true
      })
      setCards(res.data.cards)
      setSets(res.data.sets)
    } catch (err) {
      console.error(err)
    }
  }

  // Calculate scale when image loads
  useEffect(() => {
    if (imageRef.current) {
      const updateScale = () => {
        const img = imageRef.current
        if (img.naturalWidth) {
          setScale(img.width / img.naturalWidth)
        }
      }

      const img = imageRef.current
      img.addEventListener('load', updateScale)
      // Also update on resize
      window.addEventListener('resize', updateScale)

      return () => {
        img.removeEventListener('load', updateScale)
        window.removeEventListener('resize', updateScale)
      }
    }
  }, [imageUrl])

  const setColors = ['red', 'green', 'blue', 'orange', 'magenta', 'cyan']

  return (
    <div style={{ margin: '20px' }}>
      <h1>SET Finder</h1>
      <input type="file" onChange={handleFileChange} accept="image/*" />
      <button onClick={handleUpload}>Upload &amp; Analyze</button>
      <div style={{ position: 'relative', marginTop: '20px', display: 'inline-block' }}>
        {imageUrl && (
          <img
            ref={imageRef}
            src={imageUrl}
            alt="uploaded"
            id="uploaded-image"
            style={{ maxWidth: '80%', height: 'auto' }}
          />
        )}
        {/* Card overlays with bounding boxes and labels */}
        {cards.map((card) => {
          const indicesOfSets = sets
            .map((s, idx) => (s.includes(card.id) ? idx + 1 : -1))
            .filter((idx) => idx !== -1)

          const bbox = card.bbox
          return (
            <div
              key={`card-${card.id}`}
              style={{
                position: 'absolute',
                left: bbox[0] * scale,
                top: bbox[1] * scale,
                width: bbox[2] * scale,
                height: bbox[3] * scale,
                pointerEvents: 'none'
              }}
            >
              {/* Add SVG overlay for contours */}
              <svg
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none'
                }}
              >
                {card.contours && card.contours.map((contour, idx) => (
                  <path
                    key={`contour-${idx}`}
                    d={`M ${contour.map(([x, y]) => 
                      `${x * bbox[2] * scale},${y * bbox[3] * scale}`
                    ).join(' L ')} Z`}
                    fill="none"
                    stroke="black"
                    strokeWidth="5"
                    opacity="0.9"
                  />
                ))}
              </svg>

              {/* Multiple borders for different sets */}
              {indicesOfSets.map((setNumber, idx) => (
                <div
                  key={`set-border-${idx}`}
                  style={{
                    position: 'absolute',
                    top: idx * 3,
                    left: idx * 3,
                    right: -idx * 3,
                    bottom: -idx * 3,
                    border: `3px solid ${setColors[(setNumber - 1) % setColors.length]}`,
                    pointerEvents: 'none'
                  }}
                />
              ))}
              {/* Card ID and Set Numbers Label */}
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  background: 'rgba(0, 0, 0, 0.7)',
                  color: 'white',
                  padding: '2px 6px',
                  borderRadius: '0 0 4px 0',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}
              >
                Card {card.id}
                {indicesOfSets.length > 0 && (
                  <span style={{ marginLeft: '4px', fontSize: '11px' }}>
                    (Set{indicesOfSets.length > 1 ? 's' : ''}{' '}
                    {indicesOfSets.join(', ')})
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Show some textual info about sets */}
      <div style={{ marginTop: '20px' }}>
        <h2>Detected {sets.length} Sets</h2>
        {sets.map((s, idx) => (
          <div key={idx}>
            <b style={{ color: setColors[idx % setColors.length] }}>
              Set #{idx + 1}:
            </b>{' '}
            Cards {s.join(', ')}
          </div>
        ))}
      </div>
    </div>
  )
}

export default App