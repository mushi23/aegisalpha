<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;

class TwelveDataService
{
    public function getLatestOHLCV(string $symbol = 'EUR/USD', string $interval = '1min')
    {
        $apiKey = env('TWELVE_DATA_API_KEY');

        $response = Http::get("https://api.twelvedata.com/time_series", [
            'symbol'   => $symbol,
            'interval' => $interval,
            'apikey'   => $apiKey,
            'outputsize' => 2, // get last 2 candles to compute return
        ]);

        \Log::info('TwelveData response:', ['body' => $response->body()]);

        \Log::info('API KEY: ' . env('TWELVE_DATA_API_KEY'));


        $data = $response->json();

        if (!isset($data['values']) || count($data['values']) < 2) {
            return null;
        }

        // Parse and compute log return and volatility
        $latest = $data['values'][0];
        $previous = $data['values'][1];

        $latestClose = (float) $latest['close'];
        $previousClose = (float) $previous['close'];

        $logReturn = log($latestClose / $previousClose);
        $volatility = abs($logReturn); // rough est. or use stddev on longer series

        return [
            'log_return' => $logReturn,
            'volatility' => $volatility,
        ];
    }
}
