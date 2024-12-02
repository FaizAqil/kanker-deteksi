const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image) 
            .resizeNearestNeighbor([224, 224]) 
            .expandDims(0) 
            .toFloat()
            .div(tf.scalar(255)); 

        // Lakukan prediksi
        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;

        // Tentukan label berdasarkan confidence score
        const label = confidenceScore <= 50 ? 'Non-cancer' : 'Cancer';
        let suggestion;

        if (label === 'Cancer') {
            suggestion = "Segera periksa ke dokter!";
        } else {
            suggestion = "Anda sehat!";
        }

        return { label, suggestion };
    } catch (error) {
        console.error('Error during prediction:', error); // Tambahkan log kesalahan
        throw new InputError('Terjadi kesalahan dalam melakukan prediksi');
    }
}

module.exports = predictClassification;
