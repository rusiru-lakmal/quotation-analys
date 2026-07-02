import { QuotationService } from './quotation.service';
export declare class QuotationController {
    private readonly quotationService;
    constructor(quotationService: QuotationService);
    processQuotation(file: any): Promise<any>;
    getQuotations(): Promise<import("./schemas/quotation.schema").Quotation[]>;
    submitCorrection(id: string, correctedData: any): Promise<any>;
}
