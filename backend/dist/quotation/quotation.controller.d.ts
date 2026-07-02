import { QuotationService } from './quotation.service';
export declare class QuotationController {
    private readonly quotationService;
    constructor(quotationService: QuotationService);
    processPdf(filePath: string): Promise<any>;
    getQuotations(): Promise<import("./schemas/quotation.schema").Quotation[]>;
    correctAndRetrain(id: string, correctedData: any): Promise<any>;
}
