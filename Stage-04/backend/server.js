const express = require('express');
const multer = require('multer');
const fsSync = require('fs');
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const port = Number(process.env.PORT || 8000);

app.use((_request, response, next) => {
  response.setHeader('Access-Control-Allow-Origin', '*');
  response.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  response.setHeader('Access-Control-Allow-Headers', '*');
  if (_request.method === 'OPTIONS') {
    return response.sendStatus(204);
  }
  next();
});
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_request, file, callback) => {
    if (path.extname(file.originalname).toLowerCase() !== '.csv') {
      return callback(new Error('Only CSV files are supported.'));
    }
    callback(null, true);
  },
});

const modelPath = path.join(__dirname, 'predict.py');
const localPython = path.join(__dirname, '.venv', 'bin', 'python');
const pythonBin = process.env.PYTHON_BIN ||
  (fsSync.existsSync(localPython) ? localPython : 'python3');

function runModel(inputPath, orbit) {
  const outputPath = `${inputPath}_day8.csv`;

  return new Promise((resolve, reject) => {
    const model = spawn(pythonBin, [
      modelPath,
      inputPath,
      '--orbit',
      orbit,
      '--out',
      outputPath,
      '--quiet',
    ]);
    const errors = [];

    model.stderr.on('data', (chunk) => errors.push(chunk));
    model.on('error', async (error) => {
      await fs.unlink(outputPath).catch(() => {});
      reject(error);
    });
    model.on('close', async (code) => {
      if (code !== 0) {
        await fs.unlink(outputPath).catch(() => {});
        return reject(new Error(Buffer.concat(errors).toString() || `Model exited with code ${code}`));
      }

      try {
        resolve(await fs.readFile(outputPath));
      } catch (error) {
        reject(error);
      } finally {
        await fs.unlink(outputPath).catch(() => {});
      }
    });
  });
}

app.get('/health', (_request, response) => {
  response.json({ status: 'ok' });
});

app.post('/predict', upload.single('file'), async (request, response, next) => {
  if (!request.file) {
    return response.status(400).json({ error: 'Upload a CSV file using the "file" field.' });
  }

  const orbit = String(request.body.orbit || '').trim().toUpperCase();
  if (!['GEO', 'MEO'].includes(orbit)) {
    await fs.unlink(request.file.path).catch(() => {});
    return response.status(400).json({ error: 'Provide "orbit" as either "GEO" or "MEO".' });
  }

  try {
    const result = await runModel(request.file.path, orbit);
    response.type('text/csv').send(result);
  } catch (error) {
    next(error);
  } finally {
    await fs.unlink(request.file.path).catch(() => {});
  }
});

app.use((error, _request, response, _next) => {
  if (error instanceof multer.MulterError || error.message === 'Only CSV files are supported.') {
    return response.status(400).json({ error: error.message });
  }
  console.error(error);
  response.status(500).json({ error: 'Model execution failed.' });
});

app.listen(port, () => {
  console.log(`Smart Horizon backend listening on port ${port}`);
});
