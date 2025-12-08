import React, { useState } from 'react';
import { 
  Box, TextField, MenuItem, Button, FormControlLabel, 
  Autocomplete, Typography, Paper, Radio, RadioGroup, FormControl, FormLabel 
} from '@mui/material';
import { Grid } from '@mui/material';
import { BRANDS, FUEL_TYPES, TRANSMISSIONS } from '../constants';
import type { CarFormData } from '../types';
import SendIcon from '@mui/icons-material/Send';

interface PredictionFormProps {
  onSubmit: (data: CarFormData) => void;
  loading: boolean;
}

const PredictionForm: React.FC<PredictionFormProps> = ({ onSubmit, loading }) => {
  const [modelType, setModelType] = useState('catboost');

  // 初始值
  const [formData, setFormData] = useState<Omit<CarFormData, 'use_resnet'>>({
    brand: 'Toyota',
    model: 'Corolla',
    year: 2019,
    age: 6,
    milage: 45000,
    fuel_type: 'Petrol',
    engine: 1800,
    max_power: 140,
    transmission: 'Automatic',
    seats: 5,
  });

const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    const updatedData = { ...formData, [name]: value };
    if (name === 'year' && value) {
      updatedData.age = 2025 - Number(value);
    } 
    else if (name === 'age' && value) {
      updatedData.year = 2025 - Number(value);
    }

    setFormData(updatedData);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ ...formData, use_resnet: modelType === 'resnet' });
  };

  return (
    <Paper elevation={3} sx={{ p: 4, borderRadius: 3 }}>
      <Typography variant="h5" gutterBottom fontWeight="bold" color="primary">
        Vehicle Details
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Fill in the specifications below to generate a price prediction.
      </Typography>

      <form onSubmit={handleSubmit}>
        <Grid container spacing={3}>
          {/* Brand & Model */}
          <Grid size={{ xs: 12, sm: 6 }}>
            <Autocomplete
              options={BRANDS}
              value={formData.brand}
              onChange={(_, newValue) => setFormData({ ...formData, brand: newValue || '' })}
              renderInput={(params) => <TextField {...params} label="Brand" required />}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6 }}>
            <TextField 
              fullWidth label="Model" name="model" 
              value={formData.model} onChange={handleChange} 
            />
          </Grid>

          {/* Year & Age */}
          <Grid size={6}>
            <TextField 
              fullWidth type="number" label="Year" name="year" 
              value={formData.year} onChange={handleChange} 
            />
          </Grid>
          <Grid size={6}>
            <TextField 
              fullWidth type="number" label="Age" name="age" 
              value={formData.age} onChange={handleChange} 
            />
          </Grid>

          {/* Specs Row 1 */}
          <Grid size={{ xs: 12, sm: 4 }}>
            <TextField 
              fullWidth select label="Fuel Type" name="fuel_type"
              value={formData.fuel_type} onChange={handleChange}
            >
              {FUEL_TYPES.map((opt) => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
            </TextField>
          </Grid>
          <Grid size={{ xs: 12, sm: 4 }}>
            <TextField 
              fullWidth select label="Transmission" name="transmission"
              value={formData.transmission} onChange={handleChange}
            >
              {TRANSMISSIONS.map((opt) => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
            </TextField>
          </Grid>
          <Grid size={{ xs: 12, sm: 4 }}>
             <TextField 
              fullWidth type="number" label="Seats" name="seats" 
              value={formData.seats} onChange={handleChange} 
            />
          </Grid>

          {/* Specs Row 2 */}
          <Grid size={{ xs: 12, sm: 4 }}>
            <TextField 
              fullWidth type="number" label="Mileage (km)" name="milage" 
              value={formData.milage} onChange={handleChange} 
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 4 }}>
             <TextField 
              fullWidth type="number" label="Engine (CC)" name="engine" 
              value={formData.engine} onChange={handleChange} 
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 4 }}>
             <TextField 
              fullWidth type="number" label="Max Power (bhp)" name="max_power" 
              value={formData.max_power} onChange={handleChange} 
            />
          </Grid>

          {/* Model Selection */}
          <Grid size={12}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 2 }}>
              <FormControl>
                <FormLabel id="model-selection-group-label" sx={{ mb: 1, fontSize: '0.875rem' }}>
                  Prediction Model
                </FormLabel>
                <RadioGroup
                  row
                  aria-labelledby="model-selection-group-label"
                  name="model-selection"
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                >
                  <FormControlLabel 
                    value="catboost" 
                    control={<Radio />} 
                    label={
                      <Box>
                        <Typography variant="body2" fontWeight="500">Catboost</Typography>
                        <Typography variant="caption" color="text.secondary">Gradient Boosting (Detailed)</Typography>
                      </Box>
                    } 
                    sx={{ mr: 4 }}
                  />
                  <FormControlLabel 
                    value="resnet" 
                    control={<Radio />} 
                    label={
                      <Box>
                        <Typography variant="body2" fontWeight="500">Resnet</Typography>
                        <Typography variant="caption" color="text.secondary">Deep Learning (Simple)</Typography>
                      </Box>
                    } 
                  />
                </RadioGroup>
              </FormControl>
            </Box>
          </Grid>

          <Grid size={12}>
            <Button 
              type="submit" 
              variant="contained" 
              size="large" 
              fullWidth 
              disabled={loading}
              startIcon={<SendIcon />}
              sx={{ py: 1.5, fontSize: '1.1rem', fontWeight: 'bold' }}
            >
              {loading ? 'Processing...' : 'Predict Price'}
            </Button>
          </Grid>
        </Grid>
      </form>
    </Paper>
  );
};

export default PredictionForm;