"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.QuotationService = void 0;
const common_1 = require("@nestjs/common");
const mongoose_1 = require("@nestjs/mongoose");
const mongoose_2 = require("mongoose");
const child_process_1 = require("child_process");
const util_1 = require("util");
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const quotation_schema_1 = require("./schemas/quotation.schema");
const execAsync = (0, util_1.promisify)(child_process_1.exec);
let QuotationService = class QuotationService {
    quotationModel;
    constructor(quotationModel) {
        this.quotationModel = quotationModel;
    }
    async processPdfAndCompare(filePath) {
        try {
            const absoluteFilePath = path.resolve(filePath);
            const workspaceDir = path.resolve(__dirname, '../../../');
            const command = `python3 infer_and_compare.py "${absoluteFilePath}"`;
            const { stdout, stderr } = await execAsync(command, { cwd: workspaceDir });
            if (stderr) {
                console.warn('Python Pipeline Warning/Stderr:', stderr);
            }
            let extractedData;
            try {
                extractedData = JSON.parse(stdout);
            }
            catch {
                const match = stdout.match(/\{[\s\S]*\}/);
                if (match) {
                    extractedData = JSON.parse(match[0]);
                }
                else {
                    throw new Error(`Could not parse JSON from output: ${stdout}`);
                }
            }
            const newQuotation = new this.quotationModel({
                originalFile: absoluteFilePath,
                extractedData: extractedData,
                status: 'PENDING_REVIEW',
            });
            await newQuotation.save();
            return newQuotation;
        }
        catch (error) {
            throw new common_1.InternalServerErrorException(`AI Pipeline Failed: ${error.message}`);
        }
    }
    async getAllQuotations() {
        return this.quotationModel.find().sort({ createdAt: -1 }).exec();
    }
    async correctAndRetrain(id, correctedData) {
        const quotation = await this.quotationModel.findByIdAndUpdate(id, { extractedData: correctedData, status: 'VERIFIED' }, { new: true });
        if (!quotation) {
            throw new common_1.NotFoundException(`Quotation with ID ${id} not found`);
        }
        const workspaceDir = path.resolve(__dirname, '../../../');
        const trainingFile = path.join(workspaceDir, 'train_data.json');
        try {
            if (fs.existsSync(trainingFile)) {
                const fileContent = fs.readFileSync(trainingFile, 'utf-8');
                const trainData = JSON.parse(fileContent);
                if (correctedData.tokens && correctedData.ner_tags && correctedData.bboxes) {
                    trainData.push({
                        tokens: correctedData.tokens,
                        ner_tags: correctedData.ner_tags,
                        bboxes: correctedData.bboxes
                    });
                    fs.writeFileSync(trainingFile, JSON.stringify(trainData, null, 2));
                }
            }
        }
        catch (e) {
            console.error('Failed to append correction to train_data.json:', e);
        }
        return { success: true, message: 'Added to Training Dataset', quotation };
    }
};
exports.QuotationService = QuotationService;
exports.QuotationService = QuotationService = __decorate([
    (0, common_1.Injectable)(),
    __param(0, (0, mongoose_1.InjectModel)(quotation_schema_1.Quotation.name)),
    __metadata("design:paramtypes", [mongoose_2.Model])
], QuotationService);
//# sourceMappingURL=quotation.service.js.map