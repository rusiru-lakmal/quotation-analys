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
Object.defineProperty(exports, "__esModule", { value: true });
exports.QuotationSchema = exports.Quotation = void 0;
const mongoose_1 = require("@nestjs/mongoose");
const mongoose_2 = require("mongoose");
let Quotation = class Quotation extends mongoose_2.Document {
    originalFile;
    extractedData;
    status;
};
exports.Quotation = Quotation;
__decorate([
    (0, mongoose_1.Prop)({ required: true }),
    __metadata("design:type", String)
], Quotation.prototype, "originalFile", void 0);
__decorate([
    (0, mongoose_1.Prop)({ type: mongoose_2.Schema.Types.Mixed, required: true }),
    __metadata("design:type", Object)
], Quotation.prototype, "extractedData", void 0);
__decorate([
    (0, mongoose_1.Prop)({ default: 'PENDING_REVIEW' }),
    __metadata("design:type", String)
], Quotation.prototype, "status", void 0);
exports.Quotation = Quotation = __decorate([
    (0, mongoose_1.Schema)({ timestamps: true })
], Quotation);
exports.QuotationSchema = mongoose_1.SchemaFactory.createForClass(Quotation);
//# sourceMappingURL=quotation.schema.js.map