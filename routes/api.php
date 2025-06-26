<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\RegimeController;

Route::post('/predict-regime', [RegimeController::class, 'predict']);
