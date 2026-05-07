const mongoose = require("mongoose");

/**
 * Metadata for each CSV upload.
 * The raw CSV file is stored in MongoDB GridFS; gridfsId references it.
 */
const uploadSchema = new mongoose.Schema({
  userName:   { type: String,  default: "Unknown" },
  userEmail:  { type: String,  required: true, index: true },
  fileName:   { type: String,  required: true },
  rowCount:   { type: Number,  default: 0 },
  gridfsId:   { type: String,  default: null },   // ID of the file in GridFS
  mapping:    { type: mongoose.Schema.Types.Mixed, default: {} },
  stats:      { type: mongoose.Schema.Types.Mixed, default: {} },
  uploadedAt: { type: Date,    default: Date.now },
});

module.exports = mongoose.model("Upload", uploadSchema);
