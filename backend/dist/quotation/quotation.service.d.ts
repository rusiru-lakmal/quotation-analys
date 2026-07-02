import { Model } from 'mongoose';
import { Quotation } from './schemas/quotation.schema';
export declare class QuotationService {
    private quotationModel;
    constructor(quotationModel: Model<Quotation>);
    processPdfAndCompare(file: any): Promise<any>;
    getAllQuotations(): Promise<Quotation[]>;
    correctAndRetrain(id: string, correctedData: any): Promise<any>;
}
