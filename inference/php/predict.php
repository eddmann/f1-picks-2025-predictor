#!/usr/bin/env php
<?php

declare(strict_types=1);

/**
 * F1 Prediction using ONNX Runtime.
 *
 * Production workflow:
 *   1. Export ONNX models:    make export/onnx
 *   2. Export race features:  make export/features RACE=2025-24 TYPE=qualifying
 *   3. Run prediction:        php predict.php <features-file>
 *
 * Docker usage:
 *   docker run --rm -v $(pwd)/models/saved/onnx:/models f1-picks-2025-predictor-php 2025-24_qualifying.json
 */

use ORT\Model;
use ORT\Runtime;
use ORT\Tensor\Transient;

$modelsDir = getenv('MODELS_DIR') ?: __DIR__ . '/../../models/onnx';
$featuresDir = $modelsDir . '/features';
$featuresFile = $argv[1] ?? null;

if ($featuresFile === null) {
    echo "Usage: php predict.php <features-file>\n\n";
    echo "Available feature files:\n";
    foreach (glob("{$featuresDir}/*.json") ?: [] as $f) {
        echo "  - " . basename($f) . "\n";
    }
    echo "\nExport features: make export/features RACE=2025-24 TYPE=qualifying\n";
    exit(1);
}

// Resolve path
if (!str_starts_with($featuresFile, '/') && !file_exists($featuresFile)) {
    $featuresFile = file_exists("{$featuresDir}/{$featuresFile}")
        ? "{$featuresDir}/{$featuresFile}"
        : $featuresFile;
}

if (!file_exists($featuresFile)) {
    echo "Error: Features file not found: {$featuresFile}\n";
    exit(1);
}

// Load features
$data = json_decode(file_get_contents($featuresFile), true);
if (!$data || !isset($data['drivers'], $data['model_type'])) {
    echo "Error: Invalid features file\n";
    exit(1);
}

$modelType = $data['model_type'];
$modelPath = "{$modelsDir}/{$modelType}.onnx";

if (!file_exists($modelPath)) {
    echo "Error: Model not found: {$modelPath}\n";
    echo "Export with: make export/onnx\n";
    exit(1);
}

// Extract driver data
$drivers = $data['drivers'];
$features = array_map(fn($d) => array_map('floatval', $d['features']), $drivers);
$driverCodes = array_column($drivers, 'driver_code');
$teams = array_combine($driverCodes, array_column($drivers, 'team'));

// Run ONNX inference
$model = new Model('f1_ranker', $modelPath);
$runtime = new Runtime($model);

// Create input tensor - nested array auto-infers shape
$inputTensor = Transient::from($features, \ORT\Tensor::FLOAT32);

// Run inference
$outputs = $runtime->run(['features' => $inputTensor]);
$outputTensor = $outputs['variable'] ?? $outputs[array_key_first($outputs)];
$scores = $outputTensor->getData();

// Flatten if needed
if (is_array($scores[0] ?? null)) {
    $scores = array_map(fn($r) => $r[0], $scores);
}

// Build rankings
$rankings = [];
foreach ($driverCodes as $i => $driver) {
    $rankings[] = ['driver' => $driver, 'team' => $teams[$driver], 'score' => (float)$scores[$i]];
}
usort($rankings, fn($a, $b) => $b['score'] <=> $a['score']);
$top3 = array_column(array_slice($rankings, 0, 3), 'driver');

// Output
$race = $data['race_info'];
$type = strtoupper(str_replace('_', ' ', $data['prediction_type']));

echo "============================================================\n";
echo "F1 {$type} PREDICTION\n";
echo "============================================================\n";
echo "{$race['event_name']} ({$race['circuit']}) - {$race['year']} R{$race['round']}\n\n";

echo "TOP 3:\n";
foreach ($top3 as $i => $d) {
    $t = $teams[$d];
    echo "  P" . ($i + 1) . ": {$d} ({$t})\n";
}

echo "\nFULL RANKING:\n";
foreach ($rankings as $i => $r) {
    $m = $i < 3 ? ' *' : '';
    printf("  P%2d: %s (%s) %.4f%s\n", $i + 1, $r['driver'], $r['team'], $r['score'], $m);
}
echo "============================================================\n";
