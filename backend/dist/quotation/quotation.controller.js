"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.QuotationController = void 0;
const common_1 = require("@nestjs/common");
const quotation_service_1 = require("./quotation.service");
let QuotationController = class QuotationController {
    quotationService;
    constructor(quotationService) {
        this.quotationService = quotationService;
    }
    async processPdf(filePath) {
        return this.quotationService.processPdfAndCompare(filePath);
    }
    async getQuotations() {
        return this.quotationService.getAllQuotations();
    }
    async correctAndRetrain(id, correctedData) {
        return this.quotationService.correctAndRetrain(id, correctedData);
    }
};
exports.QuotationController = QuotationController;
__decorate([
    (0, common_1.Post)('process'),
    __param(0, (0, common_1.Body)('filePath')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", Promise)
], QuotationController.prototype, "processPdf", null);
__decorate([
    (0, common_1.Get)(),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", []),
    __metadata("design:returntype", Promise)
], QuotationController.prototype, "getQuotations", null);
__decorate([
    (0, common_1.Put)(':id/correct'),
    __param(0, (0, common_1.Param)('id')),
    __param(1, (0, common_1.Body)('correctedData')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String, Object]),
    __metadata("design:returntype", Promise)
], QuotationController.prototype, "correctAndRetrain", null);
exports.QuotationController = QuotationController = __decorate([
    (0, common_1.Controller)('quotation'),
    __metadata("design:paramtypes", [quotation_service_1.QuotationService])
], QuotationController);
//# sourceMappingURL=quotation.controller.js.map