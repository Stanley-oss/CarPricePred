import React from 'react';
import { 
  Paper, Typography, Box, Chip, Divider, LinearProgress, Tooltip 
} from '@mui/material';
import { Grid } from '@mui/material';
import type { PredictionResponse } from '../types';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import SpeedIcon from '@mui/icons-material/Speed';

interface ResultDisplayProps {
  data: PredictionResponse | null;
  loading: boolean;
}

const formatCurrency = (val: number) => {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
};

// 自定义范围条组件
const RangeBar = ({ min, max, low, high, p50, label, color }: any) => {
    // 保持原有代码不变
    const totalRange = max - min;
    const leftPct = ((low - min) / totalRange) * 100;
    const widthPct = ((high - low) / totalRange) * 100;
    const p50Pct = ((p50 - min) / totalRange) * 100;
  
    return (
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="caption" color="text.secondary">{label} Range</Typography>
          <Typography variant="caption" fontWeight="bold">{formatCurrency(p50)}</Typography>
        </Box>
        <Box sx={{ position: 'relative', height: 24, bgcolor: '#f0f0f0', borderRadius: 4 }}>
          <Box
            sx={{
              position: 'absolute', left: `${leftPct}%`, width: `${widthPct}%`,
              height: '100%', bgcolor: color, opacity: 0.3, borderRadius: 4,
            }}
          />
          <Box
            sx={{
              position: 'absolute', left: `${leftPct}%`, width: `${widthPct}%`,
              height: '100%', borderLeft: `2px solid ${color}`, borderRight: `2px solid ${color}`, borderRadius: 4,
            }}
          />
          <Tooltip title={`Predicted: ${formatCurrency(p50)}`}>
            <Box
              sx={{
                position: 'absolute', left: `${p50Pct}%`, top: -4, bottom: -4,
                width: 4, bgcolor: 'primary.main', borderRadius: 2, zIndex: 2, boxShadow: '0 0 5px rgba(0,0,0,0.3)'
              }}
            />
          </Tooltip>
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
          <Typography variant="caption">{formatCurrency(low)}</Typography>
          <Typography variant="caption">{formatCurrency(high)}</Typography>
        </Box>
      </Box>
    );
};

const ResultDisplay: React.FC<ResultDisplayProps> = ({ data, loading }) => {
  if (loading) {
    return (
      <Paper elevation={3} sx={{ p: 4, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Box sx={{ width: '100%', textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>Calculating Valuation...</Typography>
          <LinearProgress />
        </Box>
      </Paper>
    );
  }

  if (!data) {
    return (
      <Paper elevation={0} sx={{ p: 4, height: '100%', bgcolor: 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px dashed #ccc' }}>
        <Typography variant="body1" color="text.secondary">
          Enter vehicle details and click Predict to see the valuation.
        </Typography>
      </Paper>
    );
  }

  const isCatboost = data.model_type === 'catboost' && data.result;

  return (
    <Paper elevation={4} sx={{ p: 4, height: '100%', borderRadius: 3, position: 'relative', overflow: 'hidden' }}>
      <Box sx={{ position: 'absolute', top: 0, left: 0, width: '100%', height: 6, bgcolor: isCatboost ? 'secondary.main' : 'primary.main' }} />
      
      <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: 1.5 }}>
        PREDICTED MARKET VALUE
      </Typography>
      
      <Typography variant="h3" component="div" fontWeight="800" color="primary.main" sx={{ my: 2 }}>
        {isCatboost ? formatCurrency(data.result!.p50) : formatCurrency(data.price!)}
      </Typography>

      <Chip 
        label={(data.model_type + '+meta').toUpperCase()} 
        color={data.model_type === 'catboost' ? 'secondary' : 'primary'} 
        size="small" 
        sx={{ mb: 3 }} 
      />

      <Divider sx={{ mb: 3 }} />

      {isCatboost && data.result && (
        <Box sx={{ animation: 'fadeIn 0.5s ease-in' }}>
          {/* Chart Section */}
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrendingUpIcon fontSize="small" /> Price Analysis
          </Typography>
          
          {/* 计算绘图的全局最大最小值，防止溢出 */}
          {(() => {
            const { lo, hi, lo_raw, hi_raw, p50 } = data.result;
            const globalMin = Math.min(lo, lo_raw) * 0.95;
            const globalMax = Math.max(hi, hi_raw) * 1.05;

            return (
              <Box sx={{ my: 3 }}>
                 <RangeBar 
                    min={globalMin} max={globalMax} 
                    low={lo} high={hi} p50={p50} 
                    label="Model Confidence Interval" color="#9c27b0" 
                 />
                 <RangeBar 
                    min={globalMin} max={globalMax} 
                    low={lo_raw} high={hi_raw} p50={p50} 
                    label="Raw Data Bounds" color="#607d8b" 
                 />
              </Box>
            );
          })()}

          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid size={6}>
              <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">Confidence (WR)</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, mt: 1 }}>
                  <SpeedIcon color="action" fontSize="small"/>
                  <Typography variant="h6">{(data.result.wr * 100).toFixed(1)}%</Typography>
                </Box>
                <LinearProgress variant="determinate" value={data.result.wr * 100} sx={{ mt: 1, height: 6, borderRadius: 3 }} />
              </Paper>
            </Grid>
            <Grid size={6}>
              <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">Market Multiplier</Typography>
                <Typography variant="h6" sx={{ mt: 1 }}>{data.result.market_multiplier.toFixed(3)}x</Typography>
              </Paper>
            </Grid>
          </Grid>

          <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 2 }}>
             <Typography variant="caption" display="block" color="text.secondary">Group Key</Typography>
             <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>{data.result.group_key}</Typography>
          </Box>
        </Box>
      )}
    </Paper>
  );
};

export default ResultDisplay;