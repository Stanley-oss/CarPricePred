export interface CarFormData {
  brand: string;
  model: string;
  year: number;
  age: number;
  milage: number;
  fuel_type: string;
  engine: number;
  max_power: number;
  transmission: string;
  seats: number;
  use_resnet: boolean;
}

export interface CatboostResult {
  p50: number;
  lo: number;
  hi: number;
  wr: number;
  lo_raw: number;
  hi_raw: number;
  wr_raw: number;
  group_key: string;
  market_multiplier: number;
}

export interface PredictionResponse {
  model_type: 'catboost' | 'resnet';
  price?: number;
  result?: CatboostResult;
}