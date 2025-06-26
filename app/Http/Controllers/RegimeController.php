<?php

namespace App\Http\Controllers;

use App\Services\TwelveDataService;
use Illuminate\Http\JsonResponse;

class RegimeController extends Controller
{
    public function predict(TwelveDataService $twelve): JsonResponse
    {
        $features = $twelve->getLatestOHLCV('EUR/USD');

        if (!$features) {
            return response()->json(['error' => 'Failed to fetch OHLCV'], 500);
        }

        $response = Http::post('http://127.0.0.1:8001/predict', $features);

        return response()->json([
            'regime' => $response->json()['regime'],
            'features' => $features,
        ]);
    }
}
