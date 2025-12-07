import axios from 'axios';
import type { CarFormData, PredictionResponse } from './types';

const API_URL = 'http://localhost:8000';

export const predictPrice = async (data: CarFormData): Promise<PredictionResponse> => {
  const response = await axios.post<PredictionResponse>(`${API_URL}/predict`, data);
  return response.data;
};