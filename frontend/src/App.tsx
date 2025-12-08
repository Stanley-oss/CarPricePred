import { useState } from 'react';
import { Container, Box, Typography, createTheme, ThemeProvider, CssBaseline } from '@mui/material';
import { Grid } from '@mui/material';
import PredictionForm from './components/PredictionForm';
import ResultDisplay from './components/ResultDisplay';
import type { CarFormData, PredictionResponse } from './types';
import { predictPrice } from './api';
import DirectionsCarIcon from '@mui/icons-material/DirectionsCar';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2', // 深蓝
    },
    secondary: {
      main: '#9c27b0', // 紫色
    },
    background: {
      default: '#f4f6f8',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: { fontWeight: 700 },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: { borderRadius: 8, textTransform: 'none' },
      },
    },
    MuiPaper: {
      styleOverrides: {
        rounded: { borderRadius: 16 },
      },
    },
    MuiTextField: {
      defaultProps: { variant: 'outlined', size: 'small' }
    }
  },
});

function App() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async (data: CarFormData) => {
    setLoading(true);
    try {
      const response = await predictPrice(data);
      setResult(response);
    } catch (error) {
      console.error("Prediction failed", error);
      alert("Failed to fetch prediction. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      
      {/* 实现全屏居中 */}
      <Box sx={{ 
        minHeight: '100vh', 
        width: '100vw',             // 确保宽度占满
        display: 'flex',            // 开启 Flex 布局
        flexDirection: 'column',    // 纵向排列
        justifyContent: 'center',   // 垂直方向居中
        alignItems: 'center',       // 水平方向居中
        py: 4,                      // 保持上下内边距
        bgcolor: 'background.default' // 确保背景色应用
      }}>
        <Container maxWidth="lg">
          {/* Header */}
          <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box sx={{ p: 1.5, bgcolor: 'primary.main', color: 'white', borderRadius: 3 }}>
              <DirectionsCarIcon fontSize="large" />
            </Box>
            <Box>
              <Typography variant="h4" color="text.primary">
                Car Price Prediction System
              </Typography>
              <Typography variant="subtitle1" color="text.secondary">
                Advanced Car Price Prediction System
              </Typography>
            </Box>
          </Box>

          <Grid container spacing={4}>
            {/* Left: Form */}
            <Grid size={{ xs: 12, md: 7 }}>
              <PredictionForm onSubmit={handlePredict} loading={loading} />
            </Grid>

            {/* Right: Result */}
            <Grid size={{ xs: 12, md: 5 }}>
              <Box sx={{ height: '100%', minHeight: 400 }}>
                <ResultDisplay data={result} loading={loading} />
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;