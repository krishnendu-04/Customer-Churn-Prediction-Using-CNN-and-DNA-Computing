const mongoose = require("mongoose");

/**
 * Stores rows in chunks so each document stays well under MongoDB's 16 MB limit.
 * A large CSV is split into batches of CHUNK_SIZE rows, each saved as one document.
 */
const uploadRowsSchema = new mongoose.Schema({
  uploadId:   { type: mongoose.Schema.Types.ObjectId, ref: "Upload", required: true, index: true },
  chunkIndex: { type: Number, required: true },   // 0-based chunk number
  rows:       { type: [mongoose.Schema.Types.Mixed], default: [] },
});

module.exports = mongoose.model("UploadRows", uploadRowsSchema);
